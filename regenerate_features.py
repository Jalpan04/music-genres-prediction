from src.data_loader import get_audio_files
from src.features import process_dataset
import os

def main():
    print("Regenerating features with enhanced feature set (Tempo, Rolloff, Contrast)...")
    
    # Get all audio files
    audio_paths, labels = get_audio_files()
    
    if not audio_paths:
        print("No audio files found!")
        return
        
    print(f"Found {len(audio_paths)} tracks.")
    
    # Process and overwrite existing features.csv
    # Note: process_dataset saves to data/features.csv by default
    process_dataset(audio_paths, labels, output_csv='data/features.csv')
    
    print("Regeneration complete!")

if __name__ == "__main__":
    main()
