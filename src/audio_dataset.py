import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import random
import csv
import logging

from audio_preprocessing import AudioPreprocessor

logger = logging.getLogger(__name__)

class AudioDeepfakeDataset(Dataset):
    """Dataset for audio deepfake detection"""
    
    def __init__(
        self,
        csv_file: str,
        data_dir: str,
        split: str = 'train',
        preprocessor: Optional[AudioPreprocessor] = None,
        augment: bool = False,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ):
        self.csv_file = csv_file
        self.data_dir = Path(data_dir)
        self.split = split
        self.augment = augment and split == 'train'
        
        # Initialize preprocessor
        if preprocessor is None:
            self.preprocessor = AudioPreprocessor(augment=self.augment)
        else:
            self.preprocessor = preprocessor
            self.preprocessor.augment = self.augment
        
        # Load samples from CSV
        self.samples = self._load_samples()
        
        # Split data
        self.samples = self._split_data(train_ratio, val_ratio, test_ratio)
        
        logger.info(f"Loaded {len(self.samples)} {split} samples")
    
    def _load_samples(self) -> List[Tuple[str, int]]:
        """Load samples from CSV file"""
        samples = []
        
        try:
            with open(self.csv_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Assuming CSV has 'file' and 'label' columns
                    audio_path = self.data_dir / row['file']
                    label = 1 if row['label'].lower() == 'fake' else 0
                    
                    if audio_path.exists():
                        samples.append((str(audio_path), label))
                    else:
                        logger.warning(f"Audio file not found: {audio_path}")
        
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            
            # Fallback: scan directories
            for class_name in ['REAL', 'FAKE']:
                class_dir = self.data_dir / class_name
                if class_dir.exists():
                    label = 0 if class_name == 'REAL' else 1
                    for audio_file in class_dir.glob('*.wav'):
                        samples.append((str(audio_file), label))
        
        return samples
    
    def _split_data(self, train_ratio: float, val_ratio: float, test_ratio: float) -> List[Tuple[str, int]]:
        """Split samples into train/val/test"""
        # Shuffle with fixed seed for reproducibility
        random.seed(42)
        random.shuffle(self.samples)
        
        total = len(self.samples)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        
        if self.split == 'train':
            return self.samples[:train_end]
        elif self.split == 'val':
            return self.samples[train_end:val_end]
        elif self.split == 'test':
            return self.samples[val_end:]
        else:
            raise ValueError(f"Unknown split: {self.split}")
    
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
    csv_file: str,
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 2
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, val, and test data loaders"""
    
    # Create datasets
    train_dataset = AudioDeepfakeDataset(csv_file, data_dir, split='train', augment=True)
    val_dataset = AudioDeepfakeDataset(csv_file, data_dir, split='val', augment=False)
    test_dataset = AudioDeepfakeDataset(csv_file, data_dir, split='test', augment=False)
    
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