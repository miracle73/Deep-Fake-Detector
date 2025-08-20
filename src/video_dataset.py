import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
import random
import logging
import tempfile
import os
from google.cloud import storage

# Add logger import
logger = logging.getLogger(__name__)

class VideoDataset(Dataset):
    """Dataset for video deepfake detection - supports both local and GCS data"""
    
    def __init__(
        self,
        data_dir: str = None,
        bucket_name: str = None,
        gcs_prefix: str = "processed",
        split: str = 'train',
        max_frames: int = 16,  # Match your processed data
        frame_size: Tuple[int, int] = (224, 224),
        augment: bool = False,
        cpu_optimized: bool = True,
        train_ratio: float = 0.7,
        val_ratio: float = 0.2,
        test_ratio: float = 0.1
    ):
        self.data_dir = Path(data_dir) if data_dir else None
        self.bucket_name = bucket_name
        self.gcs_prefix = gcs_prefix
        self.split = split
        self.max_frames = max_frames
        self.frame_size = frame_size
        self.augment = augment and split == 'train'
        self.cpu_optimized = cpu_optimized
        
        # Split ratios
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        
        # Class mapping - match your folder structure (Real/Fake with capital letters)
        self.class_to_idx = {'Real': 0, 'Fake': 1}
        self.idx_to_class = {0: 'Real', 1: 'Fake'}
        
        # Initialize GCS client if using cloud storage
        self.gcs_client = None
        self.gcs_bucket = None
        if bucket_name:
            try:
                self.gcs_client = storage.Client()
                self.gcs_bucket = self.gcs_client.bucket(bucket_name)
                logger.info(f"Connected to GCS bucket: {bucket_name}")
            except Exception as e:
                logger.error(f"Failed to connect to GCS: {e}")
                raise
        
        # Load dataset
        self.samples = self._load_samples()
        
        logger.info(f"Loaded {len(self.samples)} {split} samples")
    
    def _load_samples(self) -> List[Tuple[str, int]]:
        """Load all video samples based on data source"""
        if self.bucket_name:
            return self._load_gcs_samples()
        elif self.data_dir:
            return self._load_local_samples()
        else:
            raise ValueError("Either data_dir or bucket_name must be provided")
    
    def _load_local_samples(self) -> List[Tuple[str, int]]:
        """Load samples from local directory"""
        samples = []
        
        # Check both lowercase and uppercase folder names
        class_names = ['real', 'fake', 'Real', 'Fake']
        
        for class_name in class_names:
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                continue
            
            # Map to standardized label
            if class_name.lower() == 'real':
                label = 0
            else:  # fake
                label = 1
            
            # Look for preprocessed frame files
            for frames_file in class_dir.glob('*.npz'):
                samples.append((str(frames_file), label))
        
        return self._split_samples(samples)
    
    def _load_gcs_samples(self) -> List[Tuple[str, int]]:
        """Load samples from Google Cloud Storage"""
        samples = []
        
        logger.info(f"Loading samples from gs://{self.bucket_name}/{self.gcs_prefix}/")
        
        # Load from both Real and Fake folders
        for class_name, label in [('Real', 0), ('Fake', 1)]:
            prefix = f"{self.gcs_prefix}/{class_name}/"
            
            try:
                blobs = self.gcs_client.list_blobs(self.gcs_bucket, prefix=prefix)
                class_count = 0
                
                for blob in blobs:
                    if blob.name.endswith('.npz'):
                        samples.append((blob.name, label))
                        class_count += 1
                
                logger.info(f"Found {class_count} {class_name} samples")
                
            except Exception as e:
                logger.error(f"Error loading {class_name} samples: {e}")
        
        if not samples:
            logger.error("No samples found! Check your GCS bucket and prefix.")
            raise ValueError("No samples found in GCS bucket")
        
        return self._split_samples(samples)
    
    def _split_samples(self, samples: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
        """Split samples into train/val/test based on split parameter"""
        # Shuffle samples consistently
        random.seed(42)  # For reproducible splits
        random.shuffle(samples)
        
        total_samples = len(samples)
        train_end = int(total_samples * self.train_ratio)
        val_end = train_end + int(total_samples * self.val_ratio)
        
        if self.split == 'train':
            return samples[:train_end]
        elif self.split == 'val':
            return samples[train_end:val_end]
        elif self.split == 'test':
            return samples[val_end:]
        else:
            raise ValueError(f"Unknown split: {self.split}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a video sample"""
        sample_path, label = self.samples[idx]
        
        try:
            # Load frames based on source type
            if self.bucket_name:
                frames = self._load_gcs_frames(sample_path)
            else:
                frames = self._load_local_frames(sample_path)
            
            if frames is None:
                logger.warning(f"Failed to load frames for sample {idx}")
                return self._get_dummy_sample(label)
            
            # Ensure correct number of frames
            frames = self._process_frames(frames)
            
            # Apply augmentations if enabled
            if self.augment:
                frames = self._apply_augmentations(frames)
            
            # Normalize to [0, 1]
            frames = frames.astype(np.float32) / 255.0
            
            # Convert to tensor: (channels, frames, height, width)
            frames_tensor = torch.from_numpy(frames)
            frames_tensor = frames_tensor.permute(3, 0, 1, 2)  # (C, T, H, W)
            
            # Apply ImageNet normalization
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
            frames_tensor = (frames_tensor - mean) / std
            
            return frames_tensor, label
            
        except Exception as e:
            logger.error(f"Error loading sample {idx}: {e}")
            return self._get_dummy_sample(label)
    
    def _load_local_frames(self, frames_path: str) -> Optional[np.ndarray]:
        """Load frames from local file"""
        try:
            data = np.load(frames_path)
            return data['frames']
        except Exception as e:
            logger.error(f"Error loading local frames from {frames_path}: {e}")
            return None
    
    def _load_gcs_frames(self, blob_name: str) -> Optional[np.ndarray]:
        """Load frames from GCS blob"""
        try:
            blob = self.gcs_bucket.blob(blob_name)
            
            with tempfile.NamedTemporaryFile() as tmp_file:
                blob.download_to_filename(tmp_file.name)
                data = np.load(tmp_file.name)
                return data['frames']
                
        except Exception as e:
            logger.error(f"Error loading GCS frames from {blob_name}: {e}")
            return None
    
    def _process_frames(self, frames: np.ndarray) -> np.ndarray:
        """Process frames to ensure correct dimensions"""
        # Ensure correct number of frames
        if len(frames) > self.max_frames:
            # Random temporal crop for training, center crop for validation
            if self.augment:
                start_idx = random.randint(0, len(frames) - self.max_frames)
            else:
                start_idx = (len(frames) - self.max_frames) // 2
            frames = frames[start_idx:start_idx + self.max_frames]
        elif len(frames) < self.max_frames:
            # Pad with last frame
            padding_needed = self.max_frames - len(frames)
            if len(frames) > 0:
                last_frame = frames[-1]
                padding = np.tile(last_frame[np.newaxis], (padding_needed, 1, 1, 1))
                frames = np.concatenate([frames, padding], axis=0)
            else:
                # If no frames, create dummy frames
                frames = np.zeros((self.max_frames, *self.frame_size, 3), dtype=np.uint8)
        
        return frames
    
    def _apply_augmentations(self, frames: np.ndarray) -> np.ndarray:
        """Apply video-specific augmentations"""
        # Random horizontal flip
        if random.random() > 0.5:
            frames = np.flip(frames, axis=2)  # Flip width dimension
        
        # Random brightness/contrast (simple)
        if random.random() > 0.5:
            brightness = random.uniform(0.8, 1.2)
            frames = np.clip(frames * brightness, 0, 255)
        
        return frames
    
    def _get_dummy_sample(self, label: int) -> Tuple[torch.Tensor, int]:
        """Return a dummy sample in case of loading errors"""
        dummy_frames = torch.zeros(3, self.max_frames, *self.frame_size)
        return dummy_frames, label
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for balanced training"""
        labels = [sample[1] for sample in self.samples]
        class_counts = np.bincount(labels)
        
        # Inverse frequency weighting
        total_samples = len(labels)
        weights = total_samples / (len(class_counts) * class_counts)
        
        return torch.tensor(weights, dtype=torch.float32)