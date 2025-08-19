import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import random

class VideoDataset(Dataset):
    """Dataset for video deepfake detection"""
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        max_frames: int = 30,
        frame_size: Tuple[int, int] = (224, 224),
        augment: bool = False,
        cpu_optimized: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_frames = max_frames
        self.frame_size = frame_size
        self.augment = augment and split == 'train'
        self.cpu_optimized = cpu_optimized
        
        # Class mapping
        self.class_to_idx = {'real': 0, 'fake': 1}
        self.idx_to_class = {0: 'real', 1: 'fake'}
        
        # Load dataset
        self.samples = self._load_samples()
        
        print(f"Loaded {len(self.samples)} {split} samples")
    
    def _load_samples(self) -> list:
        """Load all video samples"""
        samples = []
        
        for class_name in ['real', 'fake']:
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                continue
            
            # Look for preprocessed frame files
            for frames_file in class_dir.glob('*.npz'):
                label = self.class_to_idx[class_name]
                samples.append((str(frames_file), label))
        
        # Shuffle samples
        random.shuffle(samples)
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a video sample"""
        frames_path, label = self.samples[idx]
        
        try:
            # Load frames
            data = np.load(frames_path)
            frames = data['frames']  # Shape: (num_frames, H, W, C)
            
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
                last_frame = frames[-1]
                padding = np.tile(last_frame[np.newaxis], (padding_needed, 1, 1, 1))
                frames = np.concatenate([frames, padding], axis=0)
            
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
            # Return dummy sample
            dummy_frames = torch.zeros(3, self.max_frames, *self.frame_size)
            return dummy_frames, label
    
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