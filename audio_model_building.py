from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Conv1D, MaxPooling1D, Flatten, Bidirectional
from tensorflow.keras.optimizers import Adam

class AudioModelBuilding:
    def build_model(self, input_shape):
        """Build the model based on the specified architecture."""
        if self.config['model_type'] == 'cnn':
            # CNN model for spectrograms or MFCC features
            model = Sequential([
                Conv1D(64, 3, activation='relu', padding='same', input_shape=input_shape),
                MaxPooling1D(2),
                Conv1D(128, 3, activation='relu', padding='same'),
                MaxPooling1D(2),
                Conv1D(256, 3, activation='relu', padding='same'),
                MaxPooling1D(2),
                Flatten(),
                Dense(256, activation='relu'),
                Dropout(0.5),
                Dense(128, activation='relu'),
                Dropout(0.5),
                Dense(1, activation='sigmoid')
            ])

        elif self.config['model_type'] == 'lstm':
            # LSTM model for sequential audio features
            model = Sequential([
                Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape),
                Dropout(0.5),
                Bidirectional(LSTM(64)),
                Dropout(0.5),
                Dense(64, activation='relu'),
                Dropout(0.5),
                Dense(1, activation='sigmoid')
            ])

        elif self.config['model_type'] == 'cnn_lstm':
            # Combined CNN-LSTM model
            model = Sequential([
                Conv1D(64, 3, activation='relu', padding='same', input_shape=input_shape),
                MaxPooling1D(2),
                Conv1D(128, 3, activation='relu', padding='same'),
                MaxPooling1D(2),
                Bidirectional(LSTM(64, return_sequences=True)),
                Dropout(0.5),
                Bidirectional(LSTM(64)),
                Dropout(0.5),
                Dense(64, activation='relu'),
                Dropout(0.5),
                Dense(1, activation='sigmoid')
            ])

        else:
            raise ValueError(f"Unknown model type: {self.config['model_type']}")

        # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=self.config['learning_rate']),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        return model