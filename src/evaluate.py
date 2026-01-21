
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
import os

def calculate_metrics(y_true, y_pred, labels):
    """
    Computes classification metrics.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted')
    rec = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=labels))
    
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}

def plot_confusion_matrix(y_true, y_pred, labels, title='Confusion Matrix', filename='confusion_matrix.png'):
    """
    Plots the confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    os.makedirs('results', exist_ok=True)
    plt.savefig(f"results/{filename}")
    plt.close()

def plot_training_history(history, filename='training_history.png'):
    """
    Plots training and validation loss/accuracy.
    """
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'], label='Val Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    os.makedirs('results', exist_ok=True)
    plt.savefig(f"results/{filename}")
    plt.close()
