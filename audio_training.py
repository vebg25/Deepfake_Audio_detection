import os
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

class AudioTraining:
    def train(self, X_train, y_train, X_val=None, y_val=None, model_save_path=None):
        """Train the model on the prepared dataset."""
        # Set validation data if provided
        validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None

        # Set up callbacks
        callbacks = []

        # Early stopping to prevent overfitting
        early_stopping = EarlyStopping(
            monitor='val_loss' if validation_data else 'loss',
            patience=10,
            restore_best_weights=True
        )
        callbacks.append(early_stopping)

        # Model checkpoint to save the best model
        if model_save_path:
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            model_checkpoint = ModelCheckpoint(
                model_save_path,
                monitor='val_loss' if validation_data else 'loss',
                save_best_only=True
            )
            callbacks.append(model_checkpoint)

        # Build the model
        input_shape = X_train.shape[1:] if len(X_train.shape) > 1 else (X_train.shape[1], 1)
        self.model = self.build_model(input_shape)

        # Print model summary
        self.model.summary()

        # Train the model
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=self.config['batch_size'],
            epochs=self.config['epochs'],
            validation_data=validation_data,
            callbacks=callbacks
        )

        return self.history