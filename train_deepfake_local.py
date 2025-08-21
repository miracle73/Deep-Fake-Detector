#!/usr/bin/env python3
"""
Local training script for deepfake detection using local dataset
Train on data/processed/Fake and data/processed/Real folders
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import logging
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime
from typing import Dict, Any, Tuple, List, Optional
import random

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.video_model import VideoDeepfakeDetector

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LocalVideoDataset(Dataset):
    """Dataset for local video deepfake detection"""
    
    def __init__(
        self,
        data_dir: str = "data/processed",
        split: str = 'train',
        max_frames: int = 16,
        frame_size: Tuple[int, int] = (224, 224),
        augment: bool = False,
        train_ratio: float = 0.7,
        val_ratio: float = 0.2,
        test_ratio: float = 0.1
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_frames = max_frames
        self.frame_size = frame_size
        self.augment = augment and split == 'train'
        
        # Class mapping
        self.class_to_idx = {'Real': 0, 'Fake': 1}
        self.idx_to_class = {0: 'Real', 1: 'Fake'}
        
        # Split ratios
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        
        # Load dataset
        self.samples = self._load_samples()
        
        logger.info(f"Loaded {len(self.samples)} {split} samples")
        
        # Log class distribution
        labels = [sample[1] for sample in self.samples]
        real_count = labels.count(0)
        fake_count = labels.count(1)
        logger.info(f"  Real samples: {real_count}")
        logger.info(f"  Fake samples: {fake_count}")
    
    def _load_samples(self) -> List[Tuple[str, int]]:
        """Load all video samples from local directories"""
        samples = []
        
        # Load from Real and Fake folders
        for class_name, label in [('Real', 0), ('Fake', 1)]:
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                logger.warning(f"Directory not found: {class_dir}")
                continue
            
            # Find all .npz files
            npz_files = list(class_dir.glob('*.npz'))
            logger.info(f"Found {len(npz_files)} {class_name} .npz files")
            
            for npz_file in npz_files:
                samples.append((str(npz_file), label))
        
        if not samples:
            raise ValueError("No samples found! Check your data directory structure.")
        
        return self._split_samples(samples)
    
    def _split_samples(self, samples: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
        """Split samples into train/val/test"""
        # Shuffle samples consistently
        random.seed(42)
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
            # Load frames from .npz file
            frames = self._load_frames(sample_path)
            
            if frames is None:
                logger.warning(f"Failed to load frames for sample {idx}")
                return self._get_dummy_sample(label)
            
            # Process frames
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
    
    def _load_frames(self, frames_path: str) -> Optional[np.ndarray]:
        """Load frames from .npz file"""
        try:
            data = np.load(frames_path)
            return data['frames']
        except Exception as e:
            logger.error(f"Error loading frames from {frames_path}: {e}")
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
        
        # Random brightness/contrast
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

class LocalTrainer:
    """Local trainer for video deepfake detection"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cpu')  # Force CPU
        
        # Training parameters
        self.batch_size = config.get('batch_size', 4)
        self.epochs = config.get('epochs', 25)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.weight_decay = config.get('weight_decay', 1e-4)
        self.gradient_accumulation_steps = config.get('gradient_accumulation_steps', 4)
        
        # Model architecture
        self.frame_encoder = config.get('frame_encoder', 'mobilenet_v2')
        self.temporal_model = config.get('temporal_model', 'lstm')
        self.max_frames = config.get('max_frames', 16)
        
        logger.info(f"🖥️ Local Training Configuration:")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Batch Size: {self.batch_size}")
        logger.info(f"   Epochs: {self.epochs}")
        logger.info(f"   Learning Rate: {self.learning_rate}")
        logger.info(f"   Max Frames: {self.max_frames}")
        logger.info(f"   Frame Encoder: {self.frame_encoder}")
        logger.info(f"   Temporal Model: {self.temporal_model}")
    
    def create_model(self) -> nn.Module:
        """Create video deepfake detection model"""
        model = VideoDeepfakeDetector(
            frame_encoder=self.frame_encoder,
            temporal_model=self.temporal_model,
            num_frames=self.max_frames,
            dropout=0.3,
            cpu_optimized=True
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"📊 Model Statistics:")
        logger.info(f"   Total Parameters: {total_params:,}")
        logger.info(f"   Trainable Parameters: {trainable_params:,}")
        
        return model
    
    def create_data_loaders(self, data_dir: str) -> tuple:
        """Create data loaders for training"""
        logger.info("📹 Creating video data loaders...")
        
        # Training dataset
        train_dataset = LocalVideoDataset(
            data_dir=data_dir,
            split='train',
            max_frames=self.max_frames,
            augment=True
        )
        
        # Validation dataset
        val_dataset = LocalVideoDataset(
            data_dir=data_dir,
            split='val',
            max_frames=self.max_frames,
            augment=False
        )
        
        # Data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=False,
            drop_last=False
        )
        
        logger.info(f"   Training samples: {len(train_dataset)}")
        logger.info(f"   Validation samples: {len(val_dataset)}")
        logger.info(f"   Training batches: {len(train_loader)}")
        logger.info(f"   Validation batches: {len(val_loader)}")
        
        return train_loader, val_loader
    
    def train_epoch(self, model: nn.Module, train_loader: DataLoader, 
                   optimizer: optim.Optimizer, criterion: nn.Module, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        optimizer.zero_grad()
        
        for batch_idx, (videos, labels) in enumerate(pbar):
            videos = videos.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = model(videos)
            loss = criterion(outputs, labels)
            
            # Scale loss for gradient accumulation
            loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            # Statistics
            running_loss += loss.item() * videos.size(0) * self.gradient_accumulation_steps
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += videos.size(0)
            
            # Update progress bar
            current_acc = running_corrects.double() / total_samples
            pbar.set_postfix({
                'Loss': f'{running_loss/total_samples:.4f}',
                'Acc': f'{current_acc:.4f}'
            })
        
        # Handle remaining gradients
        if len(train_loader) % self.gradient_accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()
        
        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects.double() / total_samples
        
        return {'loss': epoch_loss, 'accuracy': epoch_acc.item()}
    
    def validate_epoch(self, model: nn.Module, val_loader: DataLoader, 
                      criterion: nn.Module, epoch: int) -> Dict[str, float]:
        """Validate for one epoch"""
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Validation {epoch+1}")
            
            for videos, labels in pbar:
                videos = videos.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = model(videos)
                loss = criterion(outputs, labels)
                
                # Statistics
                running_loss += loss.item() * videos.size(0)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)
                total_samples += videos.size(0)
                
                # Update progress bar
                current_acc = running_corrects.double() / total_samples
                pbar.set_postfix({
                    'Loss': f'{running_loss/total_samples:.4f}',
                    'Acc': f'{current_acc:.4f}'
                })
        
        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects.double() / total_samples
        
        return {'loss': epoch_loss, 'accuracy': epoch_acc.item()}
    
    def train(self, data_dir: str = "data/processed") -> tuple:
        """Main training loop"""
        logger.info("🚀 Starting local training...")
        
        # Create model
        model = self.create_model()
        model = model.to(self.device)
        
        # Create data loaders
        train_loader, val_loader = self.create_data_loaders(data_dir)
        
        # Setup training components
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Learning rate scheduler (fixed - removed verbose parameter)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        best_val_acc = 0.0
        best_model_state = None
        patience_counter = 0
        patience = 5
        
        # Training loop
        for epoch in range(self.epochs):
            logger.info(f"\n{'='*50}")
            logger.info(f"Epoch {epoch+1}/{self.epochs}")
            logger.info(f"{'='*50}")
            
            # Training phase
            train_metrics = self.train_epoch(model, train_loader, optimizer, criterion, epoch)
            
            # Validation phase
            val_metrics = self.validate_epoch(model, val_loader, criterion, epoch)
            
            # Update scheduler
            scheduler.step(val_metrics['loss'])
            current_lr = optimizer.param_groups[0]['lr']
            
            # Save history
            history['train_loss'].append(train_metrics['loss'])
            history['train_acc'].append(train_metrics['accuracy'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_acc'].append(val_metrics['accuracy'])
            history['learning_rates'].append(current_lr)
            
            # Check for improvement
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                best_model_state = model.state_dict().copy()
                patience_counter = 0
                logger.info(f"🎉 New best validation accuracy: {best_val_acc:.4f}")
            else:
                patience_counter += 1
            
            # Print epoch summary
            logger.info(f"Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
            logger.info(f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
            logger.info(f"Learning Rate: {current_lr:.6f}")
            
            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {patience} epochs without improvement")
                break
        
        # Load best model
        if best_model_state:
            model.load_state_dict(best_model_state)
        
        training_results = {
            'best_val_accuracy': best_val_acc,
            'final_train_accuracy': history['train_acc'][-1],
            'epochs_trained': len(history['train_acc']),
            'history': history
        }
        
        logger.info(f"\n✅ Training completed!")
        logger.info(f"Best validation accuracy: {best_val_acc:.4f}")
        
        return model, training_results
    
    def save_model(self, model: nn.Module, results: Dict[str, Any], 
                   output_dir: str = "models/checkpoints"):
        """Save trained model and results"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model state
        model_path = os.path.join(output_dir, "video_deepfake_detector.pth")
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': self.config,
            'results': results,
            'model_architecture': str(model)
        }, model_path)
        
        # Save training results
        results_path = os.path.join(output_dir, "training_results.json")
        with open(results_path, 'w') as f:
            # Convert numpy types for JSON serialization
            serializable_results = {}
            for key, value in results.items():
                if key == 'history':
                    serializable_results[key] = {
                        k: [float(v) for v in hist_values] 
                        for k, hist_values in value.items()
                    }
                else:
                    serializable_results[key] = float(value) if isinstance(value, np.floating) else value
            
            json.dump(serializable_results, f, indent=2)
        
        # Save model config
        config_path = os.path.join(output_dir, "model_config.json")
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        logger.info(f"💾 Model saved to {model_path}")
        logger.info(f"📊 Results saved to {results_path}")
        
        return model_path, results_path

def main():
    """Main training function"""
    
    # Configuration for local training
    config = {
        'batch_size': 2,              # Small batch for memory efficiency
        'epochs': 25,                 # As requested
        'learning_rate': 0.001,
        'max_frames': 8,
        'frame_encoder': 'mobilenet_v2',
        'temporal_model': 'lstm',
        'gradient_accumulation_steps': 4,
        'weight_decay': 1e-4
    }
    
    # Data directory
    data_dir = "data/processed"
    
    print("🎬 LOCAL VIDEO DEEPFAKE TRAINING")
    print("=" * 60)
    print(f"🌐 Environment: Local/Codespace")
    print(f"📊 Data Source: {data_dir}/")
    print(f"📦 Batch Size: {config['batch_size']}")
    print(f"⚡ Epochs: {config['epochs']}")
    print(f"🧠 Architecture: {config['frame_encoder']} + {config['temporal_model']}")
    print("=" * 60)
    
    # Check if data directory exists
    if not Path(data_dir).exists():
        logger.error(f"❌ Data directory not found: {data_dir}")
        logger.error("Make sure you have downloaded the processed data!")
        return
    
    # Check for Real and Fake folders
    real_dir = Path(data_dir) / "Real"
    fake_dir = Path(data_dir) / "Fake"
    
    if not real_dir.exists() or not fake_dir.exists():
        logger.error(f"❌ Required folders not found: {real_dir} or {fake_dir}")
        return
    
    # Count files
    real_files = len(list(real_dir.glob("*.npz")))
    fake_files = len(list(fake_dir.glob("*.npz")))
    
    logger.info(f"📊 Dataset Info:")
    logger.info(f"   Real samples: {real_files}")
    logger.info(f"   Fake samples: {fake_files}")
    logger.info(f"   Total samples: {real_files + fake_files}")
    
    try:
        # Initialize trainer
        trainer = LocalTrainer(config)
        
        # Train model
        model, results = trainer.train(data_dir)
        
        # Save model
        model_path, results_path = trainer.save_model(model, results)
        
        logger.info(f"\n🎉 SUCCESS! Training completed!")
        logger.info(f"📊 Final Results:")
        logger.info(f"   Best Validation Accuracy: {results['best_val_accuracy']:.4f}")
        logger.info(f"   Epochs Trained: {results['epochs_trained']}")
        logger.info(f"💾 Model saved to: {model_path}")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()