
import os
import kagglehub
import glob
import shutil

def download_gtzan(root_dir='./data'):
    """
    Downloads the GTZAN dataset using kagglehub.
    Returns the path where data is located.
    """
    print("Downloading GTZAN dataset via kagglehub...")
    path = kagglehub.dataset_download("andradaolteanu/gtzan-dataset-music-genre-classification")
    print(f"Dataset downloaded to: {path}")
    
    # Symlink or copy to expected directory if needed, or just return the path
    # Users code expects ./data/genres usually, but we can adapt get_audio_files
    
    # Start fresh with local data link
    target_dir = os.path.abspath(root_dir)
    if not os.path.exists(target_dir):
        # Fallback to kagglehub if local dir doesn't exist
        print("Local data not found. Downloading via kagglehub...")
        path = kagglehub.dataset_download("andradaolteanu/gtzan-dataset-music-genre-classification")
        return path
        
    return target_dir

def get_audio_files(root_dir='./data/genres'):
    """
    Returns a list of all wav files and their labels.
    If root_dir is None, downloads/fetches via kagglehub first.
    """
    if root_dir is None:
         # Check if ./data/genres exists
         if os.path.exists('./data/genres'):
             root_dir = './data/genres'
         else:
             root_dir = download_gtzan()
        
    print(f"Searching for audio files in {root_dir}...")
    
    # Kaggle dataset structure often has Data/genres_original or just genres
    possible_paths = [
        root_dir,
        os.path.join(root_dir, 'genres_original'),
        os.path.join(root_dir, 'Data', 'genres_original'),
        os.path.join(root_dir, 'genres'),
    ]
    
    dataset_path = None
    for p in possible_paths:
        if os.path.exists(p) and os.path.isdir(p):
            subdirs = [d for d in os.listdir(p) if os.path.isdir(os.path.join(p, d))]
            # GTZAN has 10 genres
            if len(subdirs) >= 10: 
                dataset_path = p
                print(f"Found genre folders in: {dataset_path}")
                break
    
    if not dataset_path:
        # Fallback: recursive search for any .wav file and assume parent folder is label
        print("Standard structure not found. Searching recursively for .wav files...")
        audio_paths = []
        labels = []
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.wav'):
                    # Check if parent is a genre (naive check)
                    parent = os.path.basename(root)
                    if parent in ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']:
                        audio_paths.append(os.path.join(root, file))
                        labels.append(parent)
        
        if audio_paths:
            print(f"Found {len(audio_paths)} audio files recursively.")
            return audio_paths, labels
        else:
            print(f"Could not find audio files in {root_dir}")
            return [], []
            
    # Standard traversal if dataset_path found
    audio_paths = []
    labels = []
    
    for genre in os.listdir(dataset_path):
        genre_dir = os.path.join(dataset_path, genre)
        if os.path.isdir(genre_dir):
            for filename in os.listdir(genre_dir):
                if filename.endswith('.wav'):
                    audio_paths.append(os.path.join(genre_dir, filename))
                    labels.append(genre)
                    
    return audio_paths, labels

if __name__ == "__main__":
    download_gtzan()
