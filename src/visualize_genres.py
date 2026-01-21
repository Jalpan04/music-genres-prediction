import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import numpy as np

def visualize_genre_waveforms(data_dir='data/genres', output_file='results/genre_waveforms_comparison.png'):
    genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 
              'jazz', 'metal', 'pop', 'reggae', 'rock']
    
    # Create a figure with subplots (5 rows, 2 columns)
    fig, axes = plt.subplots(5, 2, figsize=(15, 20))
    axes = axes.flatten()
    
    print("Generating waveforms for all genres...")
    
    for i, genre in enumerate(genres):
        # Find the first wav file for this genre
        genre_dir = os.path.join(data_dir, genre)
        if not os.path.exists(genre_dir):
            print(f"Warning: Directory {genre_dir} not found. Skipping.")
            continue
            
        files = [f for f in os.listdir(genre_dir) if f.endswith('.wav')]
        if not files:
            print(f"Warning: No wav files found in {genre_dir}. Skipping.")
            continue
            
        # Use the first file
        file_path = os.path.join(genre_dir, files[0])
        
        # Load audio (load 10 seconds to keep it clean)
        y, sr = librosa.load(file_path, duration=10)
        
        # Plot
        ax = axes[i]
        librosa.display.waveshow(y, sr=sr, ax=ax, alpha=0.8)
        ax.set_title(f'{genre.capitalize()}', fontsize=12, fontweight='bold')
        ax.label_outer()
        
    plt.suptitle('Audio Waveform Comparison by Genre (10s Sample)', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight')
    print(f"Saved genre comparison to {output_file}")

if __name__ == "__main__":
    if not os.path.exists('results'):
        os.makedirs('results')
    visualize_genre_waveforms()
