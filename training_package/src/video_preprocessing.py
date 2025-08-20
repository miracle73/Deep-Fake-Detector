import cv2
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class VideoPreprocessor:
    """Extract and preprocess frames from videos"""
    
    def __init__(
        self,
        frame_rate: float = 1.0,  # frames per second to extract
        frame_size: Tuple[int, int] = (224, 224),
        max_frames: int = 30,
        quality_threshold: float = 0.3  # Skip low-quality frames
    ):
        self.frame_rate = frame_rate
        self.frame_size = frame_size
        self.max_frames = max_frames
        self.quality_threshold = quality_threshold
    
    def extract_frames(self, video_path: str) -> Optional[np.ndarray]:
        """Extract frames from video with quality filtering"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Could not open video: {video_path}")
                return None
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            
            if duration < 1.0:  # Skip very short videos
                logger.warning(f"Video too short: {video_path} ({duration:.2f}s)")
                return None
            
            # Calculate frame sampling
            frame_interval = max(1, int(fps / self.frame_rate))
            frames_to_extract = min(self.max_frames, int(duration * self.frame_rate))
            
            frames = []
            frame_count = 0
            extracted_count = 0
            
            while len(frames) < frames_to_extract and extracted_count < total_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Sample frames at specified interval
                if frame_count % frame_interval == 0:
                    # Quality check
                    if self._is_good_quality_frame(frame):
                        # Convert BGR to RGB
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        # Resize frame
                        frame_resized = cv2.resize(frame_rgb, self.frame_size)
                        frames.append(frame_resized)
                
                frame_count += 1
                extracted_count += 1
            
            cap.release()
            
            if len(frames) == 0:
                logger.warning(f"No frames extracted from: {video_path}")
                return None
            
            # Pad or truncate to max_frames
            if len(frames) < self.max_frames:
                # Repeat last frame to reach max_frames
                last_frame = frames[-1]
                while len(frames) < self.max_frames:
                    frames.append(last_frame)
            
            # Convert to numpy array: (frames, height, width, channels)
            frames_array = np.array(frames[:self.max_frames])
            logger.info(f"Extracted {len(frames)} frames from {video_path}")
            
            return frames_array
            
        except Exception as e:
            logger.error(f"Error extracting frames from {video_path}: {e}")
            return None
    
    def _is_good_quality_frame(self, frame: np.ndarray) -> bool:
        """Check if frame meets quality threshold"""
        try:
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate Laplacian variance (measure of blur)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Normalize by image size
            normalized_var = laplacian_var / (gray.shape[0] * gray.shape[1])
            
            return normalized_var > self.quality_threshold
            
        except Exception:
            return True  # If quality check fails, accept the frame
    
    def save_frames(self, frames: np.ndarray, output_path: str):
        """Save extracted frames to compressed format"""
        try:
            np.savez_compressed(output_path, frames=frames)
            logger.info(f"Saved frames to {output_path}")
        except Exception as e:
            logger.error(f"Error saving frames to {output_path}: {e}")
    
    def load_frames(self, frames_path: str) -> Optional[np.ndarray]:
        """Load frames from saved file"""
        try:
            data = np.load(frames_path)
            return data['frames']
        except Exception as e:
            logger.error(f"Error loading frames from {frames_path}: {e}")
            return None