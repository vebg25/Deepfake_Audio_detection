import os
import tensorflow as tf

class AudioLoadModel:
    def load_model(self, model_path):
        """Load a saved model."""
        if os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
            return True
        else:
            print(f"Model file not found: {model_path}")
            return False