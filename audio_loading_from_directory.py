import os
import numpy as np

class AudioLoadingFromDirectory:
    def load_dataset_from_directory(self, base_dir, split):
      """
      Load audio files from a directory structure with debugging.

      Args:
          base_dir: Base directory containing training, validation, and testing folders
          split: One of 'training', 'validation', or 'testing'

      Returns:
          X: Features
          y: Labels (0 for real, 1 for fake)
      """
      split_dir = os.path.join(base_dir, split)
      if not os.path.exists(split_dir):
          raise ValueError(f"Directory not found: {split_dir}")

      # Get real and fake directories
      real_dir = os.path.join(split_dir, 'real')
      fake_dir = os.path.join(split_dir, 'fake')

      if not os.path.exists(real_dir) or not os.path.exists(fake_dir):
          raise ValueError(f"Missing 'real' or 'fake' directory in {split_dir}")

      # Get file lists
      real_files = [os.path.join(real_dir, f) for f in os.listdir(real_dir)
                  if f.endswith(('.wav', '.mp3'))]
      fake_files = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir)
                  if f.endswith(('.wav', '.mp3'))]

      print(f"Found {len(real_files)} real and {len(fake_files)} fake audio files in {split} set")

      # Prepare data
      X = []
      y = []

      # Debug variables
      feature_shapes = []

      # Process real files
      for i, file_path in enumerate(real_files):
          if i % 100 == 0:
              print(f"Processing real file {i}/{len(real_files)}")

          # Apply data augmentation only on training set
          apply_augmentation = self.config['use_augmentation'] and split == 'training'

          # Load and preprocess audio
          audio = self.load_audio_file(file_path)
          if audio is None:
              continue

          # Apply augmentation if enabled
          if apply_augmentation:
              audio = self.augment_audio(audio)

          # Extract features
          features = self.extract_features(audio)
          feature_shapes.append(features.shape)

          X.append(features)
          y.append(0)  # 0 for real

      # Process fake files
      for i, file_path in enumerate(fake_files):
          if i % 100 == 0:
              print(f"Processing fake file {i}/{len(fake_files)}")

          # Apply data augmentation only on training set
          apply_augmentation = self.config['use_augmentation'] and split == 'training'

          # Load and preprocess audio
          audio = self.load_audio_file(file_path)
          if audio is None:
              continue

          # Apply augmentation if enabled
          if apply_augmentation:
              audio = self.augment_audio(audio)

          # Extract features
          features = self.extract_features(audio)
          feature_shapes.append(features.shape)

          X.append(features)
          y.append(1)  # 1 for fake

      # Analyze feature shapes
      unique_shapes = set(str(shape) for shape in feature_shapes)
      print(f"Found {len(unique_shapes)} different feature shapes:")
      for shape in unique_shapes:
          count = sum(1 for s in feature_shapes if str(s) == shape)
          print(f"  - Shape {shape}: {count} samples ({count/len(feature_shapes)*100:.2f}%)")

      # Fix: use padding for consistent shapes
      if self.config['feature_type'] == 'raw':
          # For raw audio, reshape to 1D arrays
          X_padded = np.array([x.flatten() for x in X])
      else:
          # For 2D features (like MFCCs or spectrograms)
          # Find maximum dimensions
          max_time_frames = max(x.shape[0] for x in X)
          feature_dim = X[0].shape[1]

          print(f"Padding all features to shape: ({max_time_frames}, {feature_dim})")

          # Create padded array
          X_padded = np.zeros((len(X), max_time_frames, feature_dim))

          # Fill in the actual data
          for i, x in enumerate(X):
              # Copy the actual data
              X_padded[i, :x.shape[0], :] = x

      y = np.array(y)

      print(f"Final dataset shape: X={X_padded.shape}, y={y.shape}")
      return X_padded, y