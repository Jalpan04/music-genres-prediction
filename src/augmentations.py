import numpy as np
import librosa
import torch

class AudioAugmentations:
    """
    Apply augmentations to raw audio waveform.
    """
    @staticmethod
    def add_noise(y, noise_level=0.005):
        noise = np.random.randn(len(y))
        y_aug = y + noise_level * noise
        return y_aug.astype(np.float32)

    @staticmethod
    def pitch_shift(y, sr, n_steps=2.0):
        # n_steps: float, number of semitones
        # Randomly choose between -n_steps and +n_steps
        steps = np.random.uniform(-n_steps, n_steps)
        y_aug = librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)
        return y_aug

    @staticmethod
    def time_stretch(y, rate=1.0):
        # Rate > 1 speeds up, < 1 slows down
        # Randomly choose rate between 0.8 and 1.2
        r = np.random.uniform(0.8, 1.2)
        y_aug = librosa.effects.time_stretch(y, rate=r)
        return y_aug
    
    @staticmethod
    def random_apply(y, sr):
        # Randomly apply one or more augmentations
        if np.random.rand() < 0.5:
            y = AudioAugmentations.add_noise(y)
        if np.random.rand() < 0.3:
            y = AudioAugmentations.pitch_shift(y, sr)
        # Time stretch changes length, which might be tricky if we expect fixed length.
        # But our segmenter cuts afterwards or we cut before? 
        # Deep learning dataset usually expects fixed size input.
        # If we stretch, the 3s segment becomes 3.6s or 2.4s.
        # If we are segmenting *after* augmentation, it's fine. 
        # If we are augmenting *after* segmenting, we need to crop/pad.
        # For simplicity, let's skip time_stretch on segments to avoid size mismatch bugs for now, 
        # or handle it carefully. Let's skip it to be safe for "doing it properly" without shape errors.
        return y

class SpectrogramAugmentations:
    """
    Apply augmentations to Mel-Spectrogram (SpecAugment).
    Input: Tensor (channels, n_mels, time)
    """
    @staticmethod
    def freq_mask(spec, F=30, num_masks=2):
        """
        F: Maximum width of the frequency mask.
        num_masks: Number of masks to apply.
        """
        cloned = spec.clone()
        num_mel_channels = cloned.shape[1]
        
        for i in range(num_masks):
            f = np.random.randint(0, F)
            f0 = np.random.randint(0, num_mel_channels - f)
            cloned[:, f0:f0 + f, :] = 0
            
        return cloned

    @staticmethod
    def time_mask(spec, T=40, num_masks=2):
        """
        T: Maximum width of the time mask.
        """
        cloned = spec.clone()
        num_time_steps = cloned.shape[2]
        
        for i in range(num_masks):
            t = np.random.randint(0, T)
            t0 = np.random.randint(0, num_time_steps - t)
            cloned[:, :, t0:t0 + t] = 0
            
        return cloned

    @staticmethod
    def random_apply(spec):
        # Always apply light SpecAugment during training if this is called
        spec = SpectrogramAugmentations.freq_mask(spec)
        spec = SpectrogramAugmentations.time_mask(spec)
        return spec
