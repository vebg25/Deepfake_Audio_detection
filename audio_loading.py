import librosa
import numpy as np

class AudioLoading:
    def load_audio_file(self, file_path):
        """Load and preprocess a single audio file."""
        try:
            # Handle both .wav and .mp3 files
            audio, _ = librosa.load(file_path, sr=self.config['sample_rate'], mono=True)

            # Handle variable length audio files
            if len(audio) > self.config['max_audio_length']:
                audio = audio[:self.config['max_audio_length']]
            else:
                # Pad with zeros if audio is shorter than max_audio_length
                audio = np.pad(audio, (0, max(0, self.config['max_audio_length'] - len(audio))), 'constant')

            return audio
        except Exception as e:
            print(f"Error loading audio file {file_path}: {e}")
            return None