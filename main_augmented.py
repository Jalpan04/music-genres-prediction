import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd
import numpy as np

from src.data_loader import get_audio_files
from src.dataset_hybrid import HybridDataset
from src.models_hybrid import GenreHybridModel
from src.evaluate import calculate_metrics, plot_confusion_matrix, plot_training_history

def main():
    print("=== Music Genre Classification: Phase 6 (Augmented Hybrid Model) ===")
    
    # 1. Load Audio Paths
    audio_paths, labels = get_audio_files()
    if not audio_paths:
        print("Error: No audio files found.")
        return
        
    # 2. Load and Prepare Tabular Features
    features_csv = 'data/features.csv'
    if not os.path.exists(features_csv):
        print(f"Error: {features_csv} not found. Run regenerate_features.py first.")
        return
        
    df_features = pd.read_csv(features_csv)
    print(f"Loaded {len(df_features)} feature rows (Enhanced: Tempo, Rolloff, Contrast).")
        
    # 3. Label Encoding
    le = LabelEncoder()
    y_encoded = le.fit_transform(labels)
    classes = le.classes_
    class_to_idx = {c: i for i, c in enumerate(classes)}
    
    # 4. Stratified Split (File Level)
    # Using same random_state=42 ensures consistency
    paths_train, paths_test, y_train, y_test = train_test_split(
        audio_paths, labels, test_size=0.2, stratify=labels, random_state=42
    )
    
    print(f"Training Songs: {len(paths_train)}")
    print(f"Testing Songs: {len(paths_test)}")
    
    # 5. Prepare Feature DataFrames
    train_filenames = set(os.path.basename(p) for p in paths_train)
    test_filenames = set(os.path.basename(p) for p in paths_test)
    
    df_train = df_features[df_features['filename'].isin(train_filenames)].copy()
    df_test = df_features[df_features['filename'].isin(test_filenames)].copy()
    
    feature_cols = [c for c in df_train.columns if c not in ['label', 'filename']]
    print(f"Tabular Feature Dimension: {len(feature_cols)}")
    
    scaler = StandardScaler()
    df_train[feature_cols] = scaler.fit_transform(df_train[feature_cols])
    df_test[feature_cols] = scaler.transform(df_test[feature_cols])
    
    # 6. Hybrid Datasets with AUGMENTATION
    # Training set gets augment=True, which now triggers Noise, Pitch, and SpecAugment
    train_dataset = HybridDataset(
        paths_train, y_train, class_to_idx, df_train,
        sr=22050, duration=30, segment_duration=3, augment=True
    )
    
    test_dataset = HybridDataset(
        paths_test, y_test, class_to_idx, df_test,
        sr=22050, duration=30, segment_duration=3, augment=False
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # 7. Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}...")
    
    model = GenreHybridModel(
        num_classes=len(classes), 
        tabular_input_dim=len(feature_cols), 
        pretrained=True
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    # 8. Training Loop (Extended Epochs for Augmentation)
    epochs = 30
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    best_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for spec, tab, targets in train_loader:
            spec = spec.to(device)
            tab = tab.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(spec, tab)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        epoch_loss = running_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for spec, tab, targets in test_loader:
                spec = spec.to(device)
                tab = tab.to(device)
                targets = targets.to(device)
                
                outputs = model(spec, tab)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                
        val_loss /= len(test_loader)
        val_acc = correct / total
        
        history['train_loss'].append(epoch_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'results/best_augmented_model.pth')
            
    print(f"Best Segment Accuracy: {best_acc:.4f}")
    
    # 9. Voting Eval
    print("\nCalculating Song-Level Accuracy (Majority Voting)...")
    model.load_state_dict(torch.load('results/best_augmented_model.pth'))
    model.eval()
    
    song_total = len(paths_test)
    segments_per_song = 10
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for spec, tab, targets in test_loader:
            spec = spec.to(device)
            tab = tab.to(device)
            outputs = model(spec, tab)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.numpy())
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    if len(all_preds) == song_total * segments_per_song:
        final_song_preds = []
        final_song_targets = []
        
        for i in range(song_total):
            start = i * segments_per_song
            end = start + segments_per_song
            song_preds = all_preds[start:end]
            true_label = all_targets[start]
            
            counts = np.bincount(song_preds, minlength=len(classes))
            voted_label = np.argmax(counts)
            
            final_song_preds.append(voted_label)
            final_song_targets.append(true_label)
            
        acc_dict = calculate_metrics(final_song_targets, final_song_preds, classes)
        print(f"Augmented Hybrid Voting Accuracy: {acc_dict['accuracy']*100:.2f}%")
        
        plot_confusion_matrix(final_song_targets, final_song_preds, classes, title='Augmented Hybrid Confusion Matrix', filename='augmented_voting_cm.png')
    else:
        print("Warning: Prediction variance. Skipping voting.")
        
    plot_training_history(history, filename='augmented_training_history.png')

if __name__ == '__main__':
    main()
