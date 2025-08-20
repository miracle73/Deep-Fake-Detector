#!/usr/bin/env python3
"""
Vertex AI CPU-Optimized Training for Video Deepfake Detection
Designed for cost-effective training on CPU instances
Updated to work with processed data in GCS (processed/Real/ and processed/Fake/)
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from google.cloud import aiplatform
from google.cloud import storage

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.video_model import VideoDeepfakeDetector
from src.video_dataset import VideoDataset
from src.utils import setup_logging, save_model_to_gcs

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CPUOptimizedTrainer:
    """CPU-optimized trainer for video deepfake detection"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cpu')  # Force CPU for cost optimization
        
        # Training parameters optimized for CPU
        self.batch_size = config.get('batch_size', 4)  # Small batch for CPU
        self.epochs = config.get('epochs', 20)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.weight_decay = config.get('weight_decay', 1e-4)
        
        # CPU-specific optimizations
        self.num_workers = min(4, os.cpu_count())  # Limit workers for CPU
        self.gradient_accumulation_steps = config.get('gradient_accumulation_steps', 4)
        
        # Model architecture (CPU-friendly)
        self.frame_encoder = config.get('frame_encoder', 'mobilenet_v2')  # Lightweight
        self.temporal_model = config.get('temporal_model', 'lstm')  # Efficient for CPU
        self.max_frames = config.get('max_frames', 16)  # Match processed data
        
        logger.info(f"🖥️ CPU Training Configuration:")
        logger.info(f"   Batch Size: {self.batch_size}")
        logger.info(f"   Gradient Accumulation: {self.gradient_accumulation_steps}")
        logger.info(f"   Max Frames: {self.max_frames}")
        logger.info(f"   Frame Encoder: {self.frame_encoder}")
        logger.info(f"   Workers: {self.num_workers}")
    
    def create_model(self) -> nn.Module:
        """Create CPU-optimized video deepfake detection model"""
        model = VideoDeepfakeDetector(
            frame_encoder=self.frame_encoder,
            temporal_model=self.temporal_model,
            num_frames=self.max_frames,
            dropout=0.3,
            cpu_optimized=True  # Enable CPU optimizations
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"📊 Model Statistics:")
        logger.info(f"   Total Parameters: {total_params:,}")
        logger.info(f"   Trainable Parameters: {trainable_params:,}")
        
        return model
    
    def create_data_loaders(self, bucket_name: str) -> tuple:
        """Create data loaders optimized for CPU training from GCS processed data"""
        logger.info("📹 Creating video data loaders from processed GCS data...")
        
        # Training dataset - Load from processed/Real/ and processed/Fake/
        train_dataset = VideoDataset(
            bucket_name=bucket_name,
            gcs_prefix="processed",  # Points to your processed folder
            split='train',
            max_frames=self.max_frames,
            frame_size=(224, 224),
            augment=True,
            cpu_optimized=True
        )
        
        # Validation dataset
        val_dataset = VideoDataset(
            bucket_name=bucket_name,
            gcs_prefix="processed",  # Points to your processed folder
            split='val',
            max_frames=self.max_frames,
            frame_size=(224, 224),
            augment=False,
            cpu_optimized=True
        )
        
        # Data loaders with CPU optimization
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,  # Disable for CPU
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False
        )
        
        logger.info(f"   Training samples: {len(train_dataset)}")
        logger.info(f"   Validation samples: {len(val_dataset)}")
        logger.info(f"   Training batches: {len(train_loader)}")
        logger.info(f"   Validation batches: {len(val_loader)}")
        
        # Log data distribution
        try:
            class_weights = train_dataset.get_class_weights()
            logger.info(f"   Class weights: Real={class_weights[0]:.3f}, Fake={class_weights[1]:.3f}")
        except:
            logger.info("   Could not calculate class weights")
        
        return train_loader, val_loader
    
    def train_epoch(self, model: nn.Module, train_loader: DataLoader, 
                   optimizer: optim.Optimizer, criterion: nn.Module, epoch: int) -> Dict[str, float]:
        """Train for one epoch with CPU optimizations"""
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
    
    def train(self, bucket_name: str) -> tuple:
        """Main training loop"""
        logger.info("🚀 Starting CPU-optimized training...")
        logger.info(f"📊 Loading data from gs://{bucket_name}/processed/")
        
        # Create model
        model = self.create_model()
        model = model.to(self.device)
        
        # Create data loaders from GCS processed data
        train_loader, val_loader = self.create_data_loaders(bucket_name)
        
        # Setup training components
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True
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
                   output_dir: str = "/tmp/model_output"):
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

def upload_results_to_gcs(local_dir: str, bucket_name: str, gcs_prefix: str):
    """Upload training results to GCS"""
    logger.info("☁️ Uploading results to GCS...")
    
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        
        for file_path in Path(local_dir).rglob('*'):
            if file_path.is_file():
                relative_path = file_path.relative_to(local_dir)
                blob_name = f"{gcs_prefix}/{relative_path}"
                
                blob = bucket.blob(blob_name)
                blob.upload_from_filename(str(file_path))
                
                logger.info(f"✅ Uploaded: gs://{bucket_name}/{blob_name}")
        
        logger.info("✅ Results uploaded to GCS")
        
    except Exception as e:
        logger.error(f"❌ Failed to upload results: {e}")

def main():
    """Main training function for Vertex AI"""
    parser = argparse.ArgumentParser(description='CPU-Optimized Video Deepfake Training')
    
    # GCS Configuration
    parser.add_argument('--bucket-name', type=str, required=True,
                       help='GCS bucket containing processed data (processed/Real/ and processed/Fake/)')
    parser.add_argument('--output-dir', type=str, default='/tmp/model_output',
                       help='Local output directory for model files')
    parser.add_argument('--gcs-prefix', type=str, default='models/experiments',
                       help='GCS prefix for saving trained models')
    parser.add_argument('--config', type=str, help='Training configuration JSON file')
    
    # Training hyperparameters
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Training batch size (default: 4 for CPU)')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Initial learning rate')
    parser.add_argument('--max-frames', type=int, default=16,
                       help='Maximum frames per video (should match processed data)')
    parser.add_argument('--frame-encoder', type=str, default='mobilenet_v2',
                       choices=['mobilenet_v2', 'resnet18'],
                       help='Frame encoder architecture')
    parser.add_argument('--temporal-model', type=str, default='lstm',
                       choices=['lstm', 'gru', 'none'],
                       help='Temporal modeling approach')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = {
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'learning_rate': args.learning_rate,
            'max_frames': args.max_frames,
            'frame_encoder': args.frame_encoder,
            'temporal_model': args.temporal_model,
            'gradient_accumulation_steps': 4,
            'weight_decay': 1e-4
        }
    
    # Add experiment metadata
    config['experiment_id'] = datetime.now().strftime('%Y%m%d_%H%M%S')
    config['training_started_at'] = datetime.now().isoformat()
    config['bucket_name'] = args.bucket_name
    config['data_source'] = f"gs://{args.bucket_name}/processed/"
    
    # Print configuration
    print("🎬 VERTEX AI CPU-OPTIMIZED VIDEO DEEPFAKE TRAINING")
    print("=" * 60)
    print(f"🌐 Environment: Vertex AI (CPU Optimized)")
    print(f"☁️ Data Source: gs://{args.bucket_name}/processed/")
    print(f"📊 Expected Data: Real (1,186) + Fake (1,042) videos")
    print(f"🎥 Frames per Video: {args.max_frames}")
    print(f"📦 Batch Size: {args.batch_size}")
    print(f"⚡ Epochs: {args.epochs}")
    print(f"🧠 Architecture: {args.frame_encoder} + {args.temporal_model}")
    print("=" * 60)
    
    try:
        # Initialize trainer
        trainer = CPUOptimizedTrainer(config)
        
        # Train model using GCS bucket
        model, results = trainer.train(args.bucket_name)
        
        # Save model locally
        model_path, results_path = trainer.save_model(model, results, args.output_dir)
        
        # Upload to GCS
        experiment_prefix = f"{args.gcs_prefix}/{config['experiment_id']}"
        upload_results_to_gcs(args.output_dir, args.bucket_name, experiment_prefix)
        
        logger.info("✅ Training pipeline completed successfully!")
        logger.info(f"📊 Final Results:")
        logger.info(f"   Best Validation Accuracy: {results['best_val_accuracy']:.4f}")
        logger.info(f"   Epochs Trained: {results['epochs_trained']}")
        logger.info(f"   Data Used: gs://{args.bucket_name}/processed/ (Real + Fake)")
        logger.info(f"   Model Saved: gs://{args.bucket_name}/{experiment_prefix}")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()