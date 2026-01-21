
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np

from src.data_loader import get_audio_files
from src.dataset_segmented import SegmentedCNNDataset
from src.models_transfer import GenreResNet
from src.evaluate import calculate_metrics, plot_confusion_matrix, plot_training_history

def main():
    print("=== Music Genre Classification: Segmented (10x Data) ResNet ===")
    
    # 1. Load Data
    # We still get file paths. Train/Test split must happen at FILE level.
    audio_paths, labels = get_audio_files()
    if not audio_paths:
        print("Error: No audio files found.")
        return
        
    le = LabelEncoder()
    y_encoded = le.fit_transform(labels)
    classes = le.classes_
    class_to_idx = {c: i for i, c in enumerate(classes)}
    
    # Stratified Split of SONGS
    paths_train, paths_test, y_train, y_test = train_test_split(
        audio_paths, labels, test_size=0.2, stratify=labels, random_state=42
    )
    
    print(f"Training Songs: {len(paths_train)} -> Segments: ~{len(paths_train)*10}")
    print(f"Testing Songs: {len(paths_test)} -> Segments: ~{len(paths_test)*10}")
    
    # 2. Datasets
    # Augmentation enabled for training segments
    # Note: Segmentation itself is a form of augmentation (cropping), but we can also add noise/pitch/etc.
    train_dataset = SegmentedCNNDataset(
        paths_train, y_train, class_to_idx, 
        sr=22050, duration=30, segment_duration=3, augment=True
    )
    
    test_dataset = SegmentedCNNDataset(
        paths_test, y_test, class_to_idx, 
        sr=22050, duration=30, segment_duration=3, augment=False
    )
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    # 3. Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}...")
    
    model = GenreResNet(num_classes=len(classes), pretrained=True).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    # 4. Training Loop
    epochs = 20 # 10x data means more updates per epoch, so 20 epochs is plenty
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    best_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
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
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
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
            torch.save(model.state_dict(), 'results/best_segmented_model.pth')
            
    print(f"Best Segment Accuracy: {best_acc:.4f}")
    
    # 5. Analysis: Majority Vote per Song (Real World Metric)
    print("\nCalculating Song-Level Accuracy (Majority Voting)...")
    model.load_state_dict(torch.load('results/best_segmented_model.pth'))
    model.eval()
    
    song_correct = 0
    song_total = len(paths_test)
    
    # To do voting, we need to group segments by song.
    # Easiest way: re-iterate over test files manually (not using the loader shuffled/flattened) 
    # OR we assume the loader returns segments in order if shuffle=False.
    # Since shuffle=False and dataset logic is deterministic (0..9 = file0), we can chunk predictions.
    
    segments_per_song = 10 # Hardcoded to match dataset default
    
    with torch.no_grad():
        # Get all predictions for test set
        all_preds = []
        all_targets = []
        
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.numpy())
            
        # Group by 10
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        # Validate length
        expected_len = song_total * segments_per_song
        if len(all_preds) != expected_len:
             # handle edge case if some files were short/failures
             # fallback to segment acc
             print(f"Warning: Pred length {len(all_preds)} != Expected {expected_len}. Skipping voting analysis.")
        else:
            final_song_preds = []
            final_song_targets = []
            
            for i in range(song_total):
                start = i * segments_per_song
                end = start + segments_per_song
                
                song_segment_preds = all_preds[start:end]
                true_label = all_targets[start] # All segments have same label
                
                # Majority vote
                counts = np.bincount(song_segment_preds, minlength=len(classes))
                voted_label = np.argmax(counts)
                
                final_song_preds.append(voted_label)
                final_song_targets.append(true_label)
                
            song_acc = calculate_metrics(final_song_targets, final_song_preds, classes)['accuracy']
            print(f"Song-Level Voting Accuracy: {song_acc*100:.2f}%")
            
            plot_confusion_matrix(final_song_targets, final_song_preds, classes, title='Segmented Voting Confusion Matrix', filename='segmented_voting_cm.png')

    plot_training_history(history, filename='segmented_training_history.png')
    print("Segmentation Pipeline Complete.")

if __name__ == "__main__":
    main()
