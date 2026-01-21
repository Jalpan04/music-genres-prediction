
import numpy as np
import librosa

class AudioAugmentations:
    """
    Apply augmentations to raw audio waveforms.
    """
    def __init__(self, sample_rate=22050):
        self.sr = sample_rate

    def add_noise(self, data, noise_factor=0.005):
        """Adds white noise."""
        noise = np.random.randn(len(data))
        augmented_data = data + noise_factor * noise
        # Cast back to float32
        return augmented_data.astype(type(data[0]))

    def pitch_shift(self, data, n_steps=2):
        """Shifts pitch by n_steps semitones."""
        return librosa.effects.pitch_shift(data, sr=self.sr, n_steps=n_steps)

    def time_stretch(self, data, rate=1.1):
        """Stretches time. rate > 1 speeds up, rate < 1 slows down."""
        # Note: input length changes, need to handle padding/cropping in dataset
        return librosa.effects.time_stretch(data, rate=rate)

    def apply_random(self, data):
        """Applies random augmentations."""
        if np.random.random() < 0.5:
             data = self.add_noise(data)
        
        if np.random.random() < 0.3:
            steps = np.random.randint(-2, 2)
            if steps != 0:
                data = self.pitch_shift(data, steps)
                
        # Time stretch changes length which complicates fixed-size input for CNN
        # Validation typically requires fixed size. We will skip time_stretch for simplicity
        # or handle it by fixing length in dataset __getitem__.
        
        return data
