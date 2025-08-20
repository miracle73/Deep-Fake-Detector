"""
Video Deepfake Detection Package
"""

__version__ = "1.0.0"
__author__ = "Your Name"

# Make key modules easily accessible
from .video_model import VideoDeepfakeDetector
from .video_preprocessing import VideoPreprocessor
from .video_dataset import VideoDataset

# Optional: Define what gets imported with "from src import *"
__all__ = [
    'VideoDeepfakeDetector',
    'VideoPreprocessor', 
    'VideoDataset',
    'utils'
]