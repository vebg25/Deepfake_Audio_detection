import numpy as np

class AudioPrediction:
    def predict(self, audio_file):
        """
        Predict whether an audio file is genuine or spoofed.

        Args:
            audio_file: Path to the audio file

        Returns:
            score: Probability of being a spoof (0 to 1)
            prediction: Binary prediction (0 for genuine, 1 for spoof)
        """
        if self.model is None:
            raise ValueError("Model is not trained or loaded")

        # Load and preprocess audio
        audio = self.load_audio_file(audio_file)
        if audio is None:
            return None, None

        # Extract features
        features = self.extract_features(audio)

        # Preprocess features
        if self.scaler is not None and self.config['feature_type'] != 'raw':
            orig_shape = features.shape
            features_reshaped = features.reshape(-1, features.shape[-1])
            features_reshaped = self.scaler.transform(features_reshaped)
            features = features_reshaped.reshape(orig_shape)

        # Add batch dimension
        features = np.expand_dims(features, 0)

        # Make prediction
        score = self.model.predict(features)[0][0]
        prediction = 1 if score > 0.5 else 0

        return score, prediction