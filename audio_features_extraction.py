import numpy as np
import librosa

class AudioFeaturesExtraction:
    def extract_features(self, audio):
        """Extract features from audio data."""
        if self.config['feature_type'] == 'mfcc':
            # Extract MFCCs
            mfccs = librosa.feature.mfcc(
                y=audio,
                sr=self.config['sample_rate'],
                n_mfcc=self.config['n_mfcc'],
                hop_length=self.config['hop_length'],
                n_fft=self.config['n_fft']
            )
            # Add delta and delta-delta features
            delta_mfccs = librosa.feature.delta(mfccs)
            delta2_mfccs = librosa.feature.delta(mfccs, order=2)
            # Combine features and transpose
            features = np.concatenate([mfccs, delta_mfccs, delta2_mfccs])
            return features.T  # Time frames as rows, features as columns

        elif self.config['feature_type'] == 'melspectrogram':
            # Extract mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=self.config['sample_rate'],
                n_fft=self.config['n_fft'],
                hop_length=self.config['hop_length'],
                n_mels=128
            )
            # Convert to log scale
            log_mel_spec = librosa.power_to_db(mel_spec)
            return log_mel_spec.T  # Time frames as rows, features as columns

        elif self.config['feature_type'] == 'raw':
            # Return raw audio
            return audio.reshape(-1, 1)

        else:
            raise ValueError(f"Unknown feature type: {self.config['feature_type']}")