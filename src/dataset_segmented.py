
import os
import torch
import torchaudio
import librosa
import numpy as np
from torch.utils.data import Dataset
from src.augmentations import AudioAugmentations

class SegmentedCNNDataset(Dataset):
    """
    Dataset that splits each audio file into multiple segments.
    Default GTZAN: 30s. Segments: 10 x 3s.
    """
    def __init__(self, audio_paths, labels, class_to_idx, sr=22050, duration=30, segment_duration=3, augment=False):
        self.audio_paths = audio_paths
        self.labels = labels
        self.class_to_idx = class_to_idx
        self.sr = sr
        self.augment = augment
        # self.augmenter = AudioAugmentations(sr) # Removed: Startic usage
        
        self.segments_per_track = int(duration // segment_duration)
        self.segment_samples = int(sr * segment_duration)
        
        # Pre-calculate Mel Spectrogram transform
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=2048, hop_length=512, n_mels=128
        )
        self.db = torchaudio.transforms.AmplitudeToDB()

    def __len__(self):
        return len(self.audio_paths) * self.segments_per_track

    def __getitem__(self, idx):
        # Determine which file and which segment
        file_idx = idx // self.segments_per_track
        seg_idx = idx % self.segments_per_track
        
        path = self.audio_paths[file_idx]
        label_str = self.labels[file_idx]
        label = self.class_to_idx[label_str]
        
        try:
            # Load audio (efficiently if possible, but librosa loads full)
            # Offset = seg_idx * 3s
            offset = seg_idx * (self.segment_samples / self.sr)
            
            # Using librosa with offset/duration provided
            y, _ = librosa.load(path, sr=self.sr, offset=offset, duration=(self.segment_samples / self.sr))
            
            # Pad if too short (e.g. end of file)
            if len(y) < self.segment_samples:
                y = np.pad(y, (0, self.segment_samples - len(y)))
            else:
                y = y[:self.segment_samples]
                
            # Augment (only if train)
            if self.augment:
                y = AudioAugmentations.random_apply(y, self.sr)
            
            # Tensor convert
            y_tensor = torch.tensor(y, dtype=torch.float32)
            
            # Spectrogram
            spec = self.melspec(y_tensor)
            spec = self.db(spec)
            
            # Add channel dim (1, F, T)
            spec = spec.unsqueeze(0)
            
            return spec, label
            
        except Exception as e:
            # print(f"Error loading {path} segment {seg_idx}: {e}")
            return torch.zeros((1, 128, 130)), label
