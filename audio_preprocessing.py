import numpy as np
from sklearn.preprocessing import StandardScaler

class AudioPreprocessing:
    def preprocess_data(self, X_train, X_val=None, X_test=None):
        """Preprocess data by normalizing features."""
        # Reshape for standardization if not raw audio
        if self.config['feature_type'] != 'raw':
            orig_shape_train = X_train.shape
            X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])

            # Fit scaler on training data
            self.scaler = StandardScaler()
            X_train_reshaped = self.scaler.fit_transform(X_train_reshaped)
            X_train = X_train_reshaped.reshape(orig_shape_train)

            # Transform validation and test data if provided
            if X_val is not None:
                orig_shape_val = X_val.shape
                X_val_reshaped = X_val.reshape(-1, X_val.shape[-1])
                X_val_reshaped = self.scaler.transform(X_val_reshaped)
                X_val = X_val_reshaped.reshape(orig_shape_val)

            if X_test is not None:
                orig_shape_test = X_test.shape
                X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
                X_test_reshaped = self.scaler.transform(X_test_reshaped)
                X_test = X_test_reshaped.reshape(orig_shape_test)

        return X_train, X_val, X_test