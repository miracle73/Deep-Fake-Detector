import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import random
import logging

from audio_preprocessing import AudioPreprocessor

logger = logging.getLogger(__name__)

class AudioDeepfakeDataset(Dataset):
    """Dataset for audio deepfake detection"""
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        preprocessor: Optional[AudioPreprocessor] = None,
        augment: bool = False,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.augment = augment and split == 'train'
        
        # Initialize preprocessor
        if preprocessor is None:
            self.preprocessor = AudioPreprocessor(augment=self.augment)
        else:
            self.preprocessor = preprocessor
            self.preprocessor.augment = self.augment
        
        # Load samples from directories
        self.samples = self._load_samples()
        
        # Split data
        self.samples = self._split_data(train_ratio, val_ratio, test_ratio)
        
        logger.info(f"Loaded {len(self.samples)} {split} samples")
    
    def _load_samples(self) -> List[Tuple[str, int]]:
        """Load samples from REAL and FAKE directories"""
        samples = []
        
        # Define audio file extensions
        audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
        
        # Load REAL samples (label = 0) - limit to 15000
        real_dir = self.data_dir / 'REAL'
        if real_dir.exists():
            real_files = []
            for ext in audio_extensions:
                real_files.extend(list(real_dir.glob(f'*{ext}')))
            
            # Limit to 15000 files
            real_files = real_files[:15000]
            for audio_file in real_files:
                samples.append((str(audio_file), 0))
            logger.info(f"Found {len(real_files)} REAL samples (limited to 15000)")
        else:
            logger.warning(f"REAL directory not found: {real_dir}")
        
        # Load FAKE samples (label = 1) - limit to 15000
        fake_dir = self.data_dir / 'FAKE'
        if fake_dir.exists():
            fake_files = []
            for ext in audio_extensions:
                fake_files.extend(list(fake_dir.glob(f'*{ext}')))
            
            # Limit to 15000 files
            fake_files = fake_files[:15000]
            for audio_file in fake_files:
                samples.append((str(audio_file), 1))
            logger.info(f"Found {len(fake_files)} FAKE samples (limited to 15000)")
        else:
            logger.warning(f"FAKE directory not found: {fake_dir}")
        
        if not samples:
            raise ValueError(f"No audio files found in {self.data_dir}/REAL or {self.data_dir}/FAKE")
        
        logger.info(f"Total samples loaded: {len(samples)}")
        return samples
    
    def _split_data(self, train_ratio: float, val_ratio: float, test_ratio: float) -> List[Tuple[str, int]]:
        """Split samples into train/val/test while maintaining class balance"""
        
        # Separate by class
        real_samples = [s for s in self.samples if s[1] == 0]
        fake_samples = [s for s in self.samples if s[1] == 1]
        
        # Shuffle with fixed seed for reproducibility
        random.seed(42)
        random.shuffle(real_samples)
        random.shuffle(fake_samples)
        
        def split_class_samples(samples, train_ratio, val_ratio, test_ratio):
            total = len(samples)
            train_end = int(total * train_ratio)
            val_end = train_end + int(total * val_ratio)
            
            return {
                'train': samples[:train_end],
                'val': samples[train_end:val_end],
                'test': samples[val_end:]
            }
        
        # Split each class
        real_split = split_class_samples(real_samples, train_ratio, val_ratio, test_ratio)
        fake_split = split_class_samples(fake_samples, train_ratio, val_ratio, test_ratio)
        
        # Combine and shuffle the selected split
        combined_samples = real_split[self.split] + fake_split[self.split]
        random.shuffle(combined_samples)
        
        logger.info(f"{self.split} split: {len(real_split[self.split])} REAL, {len(fake_split[self.split])} FAKE")
        
        return combined_samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a single sample"""
        audio_path, label = self.samples[idx]
        
        # Process audio
        mel_spec = self.preprocessor.process_audio_file(audio_path)
        
        if mel_spec is None:
            # Return dummy sample if processing fails
            mel_spec = np.zeros((self.preprocessor.n_mels, 313))  # Default shape
            logger.warning(f"Failed to process audio: {audio_path}")
        
        # Convert to tensor and add channel dimension
        mel_tensor = torch.from_numpy(mel_spec).float().unsqueeze(0)  # (1, n_mels, time)
        
        return mel_tensor, label
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for balanced training"""
        labels = [label for _, label in self.samples]
        class_counts = np.bincount(labels)
        
        # Inverse frequency weighting
        total = len(labels)
        weights = total / (len(class_counts) * class_counts)
        
        return torch.tensor(weights, dtype=torch.float32)

def create_data_loaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 2,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, val, and test data loaders"""
    
    # Create datasets
    train_dataset = AudioDeepfakeDataset(
        data_dir, 
        split='train', 
        augment=True, 
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio
    )
    val_dataset = AudioDeepfakeDataset(
        data_dir, 
        split='val', 
        augment=False,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio
    )
    test_dataset = AudioDeepfakeDataset(
        data_dir, 
        split='test', 
        augment=False,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio
    )
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )
    
    return train_loader, val_loader, test_loader