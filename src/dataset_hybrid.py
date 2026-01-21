import os
import torch
import numpy as np
from src.dataset_segmented import SegmentedCNNDataset
from src.augmentations import SpectrogramAugmentations

class HybridDataset(SegmentedCNNDataset):
    """
    Dataset that returns (Spectrogram, TabularFeatures, Label).
    Extends SegmentedCNNDataset to reuse spectrogram generation logic.
    """
    def __init__(self, audio_paths, labels, class_to_idx, features_df, sr=22050, duration=30, segment_duration=3, augment=False):
        """
        features_df: Pandas DataFrame containing pre-scaled features. Must have a 'filename' column.
        """
        super().__init__(audio_paths, labels, class_to_idx, sr, duration, segment_duration, augment)
        self.features_df = features_df
        self.feature_map = self._build_feature_map()

    def _build_feature_map(self):
        feature_map = {}
        # Ensure we only have numeric features + filename
        # We assume 'label' is removed or we drop it here just in case
        for idx, row in self.features_df.iterrows():
            fname = row['filename']
            # Drop non-numeric info to get the feature vector
            # We assume the caller splits/scales data, so we just convert the row to tensor
            # drop 'filename' and 'label' if they exist
            feats = row.drop(['filename', 'label'], errors='ignore').values.astype(np.float32)
            feature_map[fname] = feats
        return feature_map

    def __getitem__(self, idx):
        # 1. Get Spectrogram and Label from parent
        # super().__getitem__ returns (spec, label)
        spec, label = super().__getitem__(idx)
        
        # Apply Spectrogram Augmentations (SpecAugment) if augment is True
        # Note: Audio augmentations are already applied by parent class if augment=True
        if self.augment:
            spec = SpectrogramAugmentations.random_apply(spec)
        
        # 2. Get Tabular Features
        # Determine filename from idx (logic borrowed from SegmentedCNNDataset)
        file_idx = idx // self.segments_per_track
        path = self.audio_paths[file_idx]
        fname = os.path.basename(path)
        
        if fname in self.feature_map:
            tabular = self.feature_map[fname]
        else:
            # Fallback (should not happen if data is aligned)
            # Find feature size from first entry
            feat_dim = len(next(iter(self.feature_map.values())))
            tabular = np.zeros(feat_dim, dtype=np.float32)
            
        tabular = torch.tensor(tabular, dtype=torch.float32)
        
        return spec, tabular, label
