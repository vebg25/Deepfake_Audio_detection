import os
import matplotlib.pyplot as plt
from audio_deepfake_detector import AudioDeepfakeDetector

# Example usage for the custom dataset structure
def main():
    # Example configuration
    config = {
        'feature_type': 'mfcc',
        'model_type': 'cnn_lstm',
        'n_mfcc': 40,
        'batch_size': 32,
        'epochs': 30,
        'learning_rate': 0.001,
        'use_augmentation': True  # Set to False if you don't want augmentation
    }

    # Initialize detector
    detector = AudioDeepfakeDetector(config)

    # Set path to your dataset
    dataset_dir = "/content/for-2seconds"  # Replace with your actual dataset path

    # Load datasets from your directory structure
    print("Loading training dataset...")
    X_train, y_train = detector.load_dataset_from_directory(dataset_dir, 'training')

    print("Loading validation dataset...")
    X_val, y_val = detector.load_dataset_from_directory(dataset_dir, 'validation')

    print("Loading testing dataset...")
    X_test, y_test = detector.load_dataset_from_directory(dataset_dir, 'testing')

    print(f"Train set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}")

    # Preprocess data
    print("Preprocessing data...")
    X_train, X_val, X_test = detector.preprocess_data(X_train, X_val, X_test)

    # Create model output directory
    os.makedirs("models", exist_ok=True)

    # Train model
    print("Training model...")
    detector.train(X_train, y_train, X_val, y_val, model_save_path="models/deepfake_detector.h5")

    # Evaluate model
    print("Evaluating model...")
    results = detector.evaluate(X_test, y_test)

    # Plot training history
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(detector.history.history['loss'], label='Training Loss')
    plt.plot(detector.history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(detector.history.history['accuracy'], label='Training Accuracy')
    plt.plot(detector.history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')

    # # Example of prediction on a new file
    # print("\nExample prediction:")
    # test_file = "path/to/test_audio.wav"  # Replace with path to a test file
    # if os.path.exists(test_file):
    #     score, prediction = detector.predict(test_file)
    #     print(f"File: {test_file}")
    #     print(f"Deepfake probability score: {score:.4f}")
    #     print(f"Prediction: {'FAKE' if prediction == 1 else 'REAL'}")

if __name__ == "__main__":
    main()