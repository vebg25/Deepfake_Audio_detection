import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

# Import all the necessary classes
from audio_loading import AudioLoading
from audio_features_extraction import AudioFeaturesExtraction
from audio_augmentation import AudioAugmentation
from audio_loading_from_directory import AudioLoadingFromDirectory
from audio_preprocessing import AudioPreprocessing
from audio_model_building import AudioModelBuilding
from audio_training import AudioTraining
from audio_evaluation import AudioEvaluation
from audio_load_model import AudioLoadModel
from audio_prediction import AudioPrediction

class AudioDeepfakeDetector(AudioLoading, AudioFeaturesExtraction, AudioAugmentation, 
                           AudioLoadingFromDirectory, AudioPreprocessing, 
                           AudioModelBuilding, AudioTraining, AudioEvaluation, 
                           AudioLoadModel, AudioPrediction):
    def __init__(self, config=None):
        """Initialize the audio deepfake detector with configuration."""
        self.default_config = {
            'sample_rate': 16000,
            'n_mfcc': 40,
            'hop_length': 512,
            'n_fft': 1024,
            'feature_type': 'mfcc',  # Options: 'mfcc', 'melspectrogram', 'raw'
            'model_type': 'cnn_lstm',  # Options: 'cnn', 'lstm', 'cnn_lstm'
            'max_audio_length': 64000,  # 4 seconds at 16kHz
            'batch_size': 32,
            'epochs': 50,
            'learning_rate': 0.001,
            'use_augmentation': True
        }

        self.config = self.default_config.copy()
        if config:
            self.config.update(config)

        self.scaler = None
        self.model = None
        self.history = None