
import os
import librosa
import numpy as np
import pandas as pd
import warnings

def extract_features(file_path, sr=22050):
    """
    Extracts audio features from a given file path.
    Features: MFCC, Chroma, Spectral Centroid, Bandwidth, ZCR, RMS.
    Aggregates frame-level features using mean and variance.
    """
    try:
        # Load audio
        y, sr = librosa.load(file_path, sr=sr)
        
        # Features
        # Chroma
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma_stft)
        chroma_var = np.var(chroma_stft)
        
        # RMS Energy
        rms = librosa.feature.rms(y=y)
        rms_mean = np.mean(rms)
        rms_var = np.var(rms)
        
        # Spectral Centroid
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        cent_mean = np.mean(cent)
        cent_var = np.var(cent)
        
        # Spectral Bandwidth
        bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        bw_mean = np.mean(bw)
        bw_var = np.var(bw)
        
        # Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zcr)
        zcr_var = np.var(zcr)
        
        # Harmony and Perceptr - skipping for speed/simplicity unless requested, 
        # but prompt asked for "meaningful audio features including..." list which I have covered.
        # Adding MFCCs (usually 20)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        
        features = {
            'chroma_stft_mean': chroma_mean, 'chroma_stft_var': chroma_var,
            'rms_mean': rms_mean, 'rms_var': rms_var,
            'spectral_centroid_mean': cent_mean, 'spectral_centroid_var': cent_var,
            'spectral_bandwidth_mean': bw_mean, 'spectral_bandwidth_var': bw_var,
            'zero_crossing_rate_mean': zcr_mean, 'zero_crossing_rate_var': zcr_var,
        }
        
        for i in range(1, 21):
            features[f'mfcc{i}_mean'] = np.mean(mfcc[i-1])
            features[f'mfcc{i}_var'] = np.var(mfcc[i-1])
            
        return features
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def process_dataset(audio_paths, labels, output_csv='data/features.csv'):
    """
    Iterates over audio files, extracts features, and saves to CSV.
    """
    print("Starting feature extraction...")
    data = []
    total = len(audio_paths)
    
    for i, (path, label) in enumerate(zip(audio_paths, labels)):
        if i % 100 == 0:
            print(f"Processing {i}/{total}")
        
        feats = extract_features(path)
        if feats:
            feats['label'] = label
            feats['filename'] = os.path.basename(path)
            data.append(feats)
            
    df = pd.DataFrame(data)
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Feature extraction complete. Saved to {output_csv}")
    return df

if __name__ == "__main__":
    # Test on a single file if available or nothing
    pass
