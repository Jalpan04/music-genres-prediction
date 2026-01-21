
import os
import torch
import torchaudio
import librosa
import numpy as np
from torch.utils.data import Dataset
from src.augmentations import AudioAugmentations

class CNNDataset(Dataset):
    def __init__(self, audio_paths, labels, class_to_idx, sr=22050, duration=3, augment=False):
        self.audio_paths = audio_paths
        self.labels = labels
        self.class_to_idx = class_to_idx
        self.sr = sr
        self.duration = duration
        self.target_len = sr * duration
        self.augment = augment
        self.augmenter = AudioAugmentations(sr)
        
        # Transformations
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=2048,
            hop_length=512,
            n_mels=128
        )
        self.db = torchaudio.transforms.AmplitudeToDB()

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        path = self.audio_paths[idx]
        label_str = self.labels[idx]
        label = self.class_to_idx[label_str]
        
        try:
            # Load audio
            y, _ = librosa.load(path, sr=self.sr, duration=self.duration)
            
            # Pad or truncate to ensure fixed length
            if len(y) < self.target_len:
                y = np.pad(y, (0, self.target_len - len(y)))
            else:
                y = y[:self.target_len]
                
            # Augment
            if self.augment:
                y = self.augmenter.apply_random(y)
            
            # Convert to Tensor
            y_tensor = torch.tensor(y, dtype=torch.float32)
            
            # Generate Mel-Spectrogram
            # librosa returns (n_samples,), generic MelSpec expects (..., n_samples)
            # torchaudio MelSpectrogram expects tensor
            spec = self.melspec(y_tensor)
            spec = self.db(spec)
            
            # Add channel dimension (1, n_mels, time)
            spec = spec.unsqueeze(0)
            
            return spec, label
            
        except Exception as e:
            print(f"Error loading {path}: {e}")
            # Return a dummy tensor to avoid crashing
            return torch.zeros((1, 128, 130)), label
