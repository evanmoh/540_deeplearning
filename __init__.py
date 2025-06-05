"""
Lung Nodule Detection Scripts Package
Contains data processing, feature extraction, and modeling scripts
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Import main classes for easy access
from .make_dataset import DatasetCreator
from .build_features import FeatureExtractor  
from .model import ModelTrainer, ModelPredictor

__all__ = [
    'DatasetCreator',
    'FeatureExtractor', 
    'ModelTrainer',
    'ModelPredictor'
]
