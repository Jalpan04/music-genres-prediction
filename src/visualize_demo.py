import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

def generate_visuals(file_path, output_dir='results/visuals'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    y, sr = librosa.load(file_path, duration=10) # 10s is enough for demo
    
    # 1. Mel-Spectrogram (The "Visual" input for ResNet)
    plt.figure(figsize=(10, 4))
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-Spectrogram (ResNet Input)')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/demo_melspectrogram.png')
    plt.close()
    
    # 2. Waveform (Raw Audio)
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(y, sr=sr, color='blue')
    plt.title('Audio Waveform')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/demo_waveform.png')
    plt.close()
    
    # 3. Chromagram (Harmonic Content - Feature)
    plt.figure(figsize=(10, 4))
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    librosa.display.specshow(chroma, y_axis='chroma', x_axis='time')
    plt.colorbar()
    plt.title('Chromagram (Pitch Classes)')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/demo_chromagram.png')
    plt.close()
    
    # 4. MFCCs (Timbral Content - Feature)
    plt.figure(figsize=(10, 4))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    librosa.display.specshow(mfcc, x_axis='time')
    plt.colorbar()
    plt.title('MFCC (Timbre Coefficients)')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/demo_mfcc.png')
    plt.close()
    
    # 5. Spectral Rolloff & Centroid (Shape Features)
    plt.figure(figsize=(10, 4))
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    times = librosa.times_like(rolloff)
    
    # Normalize for plotting on same scale roughly
    plt.semilogy(times, rolloff, label='Spectral Rolloff', color='red', alpha=0.9)
    plt.semilogy(times, cent, label='Spectral Centroid', color='green', alpha=0.6)
    plt.legend()
    plt.title('Spectral Rolloff & Centroid (Temporal Features)')
    plt.xlabel('Time (s)')
    plt.ylabel('Hz')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/demo_spectral_features.png')
    plt.close()
    
    print(f"Visuals saved to {output_dir}")

if __name__ == "__main__":
    # Find a sample file (e.g., from rock)
    genre = 'rock'
    base_path = 'data/genres'
    
    # Walk to find a file
    target_file = None
    for root, dirs, files in os.walk(base_path):
        for f in files:
            if f.endswith('.wav') and genre in root:
                target_file = os.path.join(root, f)
                break
        if target_file: break
            
    if not target_file:
        # Fallback to any file
        for root, dirs, files in os.walk(base_path):
            for f in files:
                if f.endswith('.wav'):
                    target_file = os.path.join(root, f)
                    break
            if target_file: break
            
    if target_file:
        print(f"Generating visuals for: {target_file}")
        generate_visuals(target_file)
    else:
        print("No audio files found to visualize.")
