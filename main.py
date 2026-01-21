
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

from src.data_loader import get_audio_files, download_gtzan
from src.features import process_dataset
from src.models_classical import train_knn, train_dt, train_svm
from src.models_neural import train_neural_network
from src.evaluate import calculate_metrics, plot_confusion_matrix, plot_training_history

import warnings
warnings.filterwarnings('ignore')

def main():
    print("=== Music Genre Classification System ===")
    
    # 1. Data Prep
    FEATURES_FILE = 'data/features.csv'
    
    if os.path.exists(FEATURES_FILE):
        print(f"Loading features from {FEATURES_FILE}...")
        df = pd.read_csv(FEATURES_FILE)
    else:
        print("Features not found. Checking dataset...")
        # data_loader.get_audio_files handles download now if needed
        audio_paths, labels = get_audio_files()
        
        if not audio_paths:
            print("Error: No audio files found.")
            return
            
        df = process_dataset(audio_paths, labels, output_csv=FEATURES_FILE)
    
    # Preprocessing
    X = df.drop(['label', 'filename'], axis=1).values
    y = df['label'].values
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    classes = le.classes_
    print(f"Classes: {classes}")
    
    # Train/Test Split (Stratified)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, stratify=y, random_state=42)
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    
    # 2. Classical Models
    print("\n--- Training Classical Models ---")
    
    # KNN
    knn_model = train_knn(X_train_scaled, y_train)
    y_pred_knn = knn_model.predict(X_test_scaled)
    print("\nKNN Results:")
    results['KNN'] = calculate_metrics(y_test, y_pred_knn, classes)
    plot_confusion_matrix(y_test, y_pred_knn, classes, title='KNN Confusion Matrix', filename='knn_confusion_matrix.png')
    
    # Decision Tree
    dt_model = train_dt(X_train_scaled, y_train)
    y_pred_dt = dt_model.predict(X_test_scaled)
    print("\nDecision Tree Results:")
    results['Decision Tree'] = calculate_metrics(y_test, y_pred_dt, classes)
    plot_confusion_matrix(y_test, y_pred_dt, classes, title='DT Confusion Matrix', filename='dt_confusion_matrix.png')
    
    # SVM
    svm_model = train_svm(X_train_scaled, y_train)
    y_pred_svm = svm_model.predict(X_test_scaled)
    print("\nSVM Results:")
    results['SVM'] = calculate_metrics(y_test, y_pred_svm, classes)
    plot_confusion_matrix(y_test, y_pred_svm, classes, title='SVM Confusion Matrix', filename='svm_confusion_matrix.png')
    
    # 3. Neural Network
    print("\n--- Training Neural Network ---")
    input_dim = X_train_scaled.shape[1]
    nn_model, history = train_neural_network(X_train_scaled, y_train, X_test_scaled, y_test, input_dim, num_classes=len(classes))
    plot_training_history(history, filename='mlp_training_history.png')
    
    # Evaluate NN
    # Use CPU for final prediction or keeping it on device? 
    # train_neural_network usually leaves model on device.
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nn_model.eval()
    with torch.no_grad():
        inputs = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
        outputs = nn_model(inputs)
        _, y_pred_nn = torch.max(outputs, 1)
        y_pred_nn = y_pred_nn.cpu().numpy()
        
    print("\nNeural Network Results:")
    results['Neural Network'] = calculate_metrics(y_test, y_pred_nn, classes)
    plot_confusion_matrix(y_test, y_pred_nn, classes, title='NN Confusion Matrix', filename='nn_confusion_matrix.png')
    
    # 4. Summary
    print("\n--- Final Comparison ---")
    summary_df = pd.DataFrame(results).T
    print(summary_df)
    summary_df.to_csv('results/model_comparison.csv')
    
    print("\nResearch Pipeline Complete. Results saved to 'results/' directory.")

if __name__ == '__main__':
    main()
