import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from pathlib import Path
import json
from typing import Tuple, Optional, Dict, List
import random

class DeepfakeDataset(Dataset):
    """Dataset class for deepfake detection"""
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        preprocessor: Optional[object] = None,
        augment: bool = False,
        balance_classes: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.preprocessor = preprocessor
        self.augment = augment and (split == "train")
        
        # Class mapping
        self.class_to_idx = {"real": 0, "fake": 1}
        self.idx_to_class = {0: "real", 1: "fake"}
        
        # Load dataset
        self.samples = self._load_samples()
        
        # Balance classes if requested
        if balance_classes and split == "train":
            self.samples = self._balance_classes(self.samples)
        
        print(f"Loaded {len(self.samples)} samples for {split} split")
        self._print_class_distribution()
    
    def _load_samples(self) -> List[Tuple[str, int]]:
        """Load all samples from the dataset directory"""
        samples = []
        
        # Path to split directory
        split_dir = self.data_dir / "processed" / self.split
        
        if not split_dir.exists():
            raise ValueError(f"Split directory not found: {split_dir}")
        
        # Load samples from each class
        for class_name in ["real", "fake"]:
            class_dir = split_dir / class_name
            
            if not class_dir.exists():
                print(f"Warning: Class directory not found: {class_dir}")
                continue
            
            # Get all image files
            for ext in ["*.jpg", "*.jpeg", "*.png"]:
                for img_path in class_dir.glob(ext):
                    label = self.class_to_idx[class_name]
                    samples.append((str(img_path), label))
        
        if not samples:
            raise ValueError(f"No samples found in {split_dir}")
        
        # Shuffle samples
        random.shuffle(samples)
        return samples
    
    def _balance_classes(self, samples: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
        """Balance classes by undersampling majority class"""
        # Separate by class
        real_samples = [s for s in samples if s[1] == 0]
        fake_samples = [s for s in samples if s[1] == 1]
        
        # Find minimum count
        min_count = min(len(real_samples), len(fake_samples))
        
        # Undersample to balance
        balanced_samples = []
        balanced_samples.extend(random.sample(real_samples, min_count))
        balanced_samples.extend(random.sample(fake_samples, min_count))
        
        # Shuffle
        random.shuffle(balanced_samples)
        
        print(f"Balanced dataset: {min_count} real, {min_count} fake")
        return balanced_samples
    
    def _print_class_distribution(self):
        """Print class distribution"""
        real_count = sum(1 for _, label in self.samples if label == 0)
        fake_count = sum(1 for _, label in self.samples if label == 1)
        
        print(f"Class distribution - Real: {real_count}, Fake: {fake_count}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a single sample"""
        img_path, label = self.samples[idx]
        
        try:
            # Load image
            image = Image.open(img_path).convert('RGB')
            
            # Apply preprocessing
            if self.preprocessor:
                image = self.preprocessor.preprocess_image(image, augment=self.augment)
            else:
                # Default preprocessing
                import torchvision.transforms as transforms
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ])
                image = transform(image)
            
            return image, label
        
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy sample
            dummy_image = torch.zeros(3, 224, 224)
            return dummy_image, label
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for loss function"""
        real_count = sum(1 for _, label in self.samples if label == 0)
        fake_count = sum(1 for _, label in self.samples if label == 1)
        
        total = real_count + fake_count
        weights = torch.tensor([total / (2 * real_count), total / (2 * fake_count)])
        
        return weights

def create_data_loaders(
    data_dir: str,
    preprocessor: object,
    batch_size: int = 32,
    num_workers: int = 4,
    balance_classes: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create data loaders for train, validation, and test sets"""
    
    # Create datasets
    train_dataset = DeepfakeDataset(
        data_dir=data_dir,
        split="train",
        preprocessor=preprocessor,
        augment=True,
        balance_classes=balance_classes
    )
    
    val_dataset = DeepfakeDataset(
        data_dir=data_dir,
        split="val",
        preprocessor=preprocessor,
        augment=False,
        balance_classes=False
    )
    
    test_dataset = DeepfakeDataset(
        data_dir=data_dir,
        split="test",
        preprocessor=preprocessor,
        augment=False,
        balance_classes=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader