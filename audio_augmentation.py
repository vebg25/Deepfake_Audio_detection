import numpy as np
import librosa

class AudioAugmentation:
    def augment_audio(self, audio):
        """Apply random augmentations to audio."""
        if not self.config['use_augmentation']:
            return audio

        # Randomly choose augmentation type
        aug_type = np.random.choice(['noise', 'stretch', 'shift', 'pitch', 'none'])

        if aug_type == 'noise':
            # Add random noise
            noise_level = np.random.uniform(0.001, 0.01)
            noise = np.random.normal(0, noise_level, len(audio))
            return audio + noise

        elif aug_type == 'stretch':
            # Time stretch
            rate = np.random.uniform(0.8, 1.2)
            return librosa.effects.time_stretch(audio, rate=rate)

        elif aug_type == 'shift':
            # Time shift
            shift = np.random.randint(-1000, 1000)
            return np.roll(audio, shift)

        elif aug_type == 'pitch':
            # Pitch shift
            n_steps = np.random.uniform(-2, 2)
            return librosa.effects.pitch_shift(audio, sr=self.config['sample_rate'], n_steps=n_steps)

        else:
            # No augmentation
            return audio