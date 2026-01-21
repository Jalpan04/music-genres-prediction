
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np

from src.data_loader import get_audio_files
from src.dataset_cnn import CNNDataset
from src.models_transfer import GenreResNet
from src.evaluate import calculate_metrics, plot_confusion_matrix, plot_training_history

def main():
    print("=== Music Genre Classification: Transfer Learning (ResNet18) ===")
    
    # 1. Load Data
    audio_paths, labels = get_audio_files()
    if not audio_paths:
        print("Error: No audio files found.")
        return
        
    le = LabelEncoder()
    y_encoded = le.fit_transform(labels)
    classes = le.classes_
    class_to_idx = {c: i for i, c in enumerate(classes)}
    
    paths_train, paths_test, y_train, y_test = train_test_split(
        audio_paths, labels, test_size=0.2, stratify=labels, random_state=42
    )
    
    # 2. Datasets
    # Augmentation enabled for training
    train_dataset = CNNDataset(paths_train, y_train, class_to_idx, augment=True)
    test_dataset = CNNDataset(paths_test, y_test, class_to_idx, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # 3. Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}...")
    
    model = GenreResNet(num_classes=len(classes), pretrained=True).to(device)
    
    criterion = nn.CrossEntropyLoss()
    # Lower learning rate for fine-tuning
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    # 4. Training Loop
    epochs = 20 # Transfer learning converges faster
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
            torch.save(model.state_dict(), 'results/best_resnet_model.pth')
            
    print(f"Best Neural Network Accuracy: {best_acc:.4f}")
    
    # 5. Evaluation
    model.load_state_dict(torch.load('results/best_resnet_model.pth'))
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.numpy())
            
    plot_training_history(history, filename='resnet_training_history.png')
    
    print("\nResNet Classification Report:")
    calculate_metrics(all_targets, all_preds, classes)
    plot_confusion_matrix(all_targets, all_preds, classes, title='ResNet Confusion Matrix', filename='resnet_confusion_matrix.png')
    
    print("Transfer Learning Complete.")

if __name__ == "__main__":
    main()
