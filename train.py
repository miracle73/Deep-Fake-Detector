#!/usr/bin/env python3
"""
Train Audio Deepfake Detection Model with Automatic Checkpointing
Saves model after each epoch and resumes from the last checkpoint
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import json
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from audio_model import AudioDeepfakeDetector
from audio_dataset import create_data_loaders
from audio_preprocessing import AudioPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Trainer:
    """Audio deepfake detection trainer with automatic checkpointing"""
    
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Training parameters
        self.epochs = config.get('epochs', 50)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.batch_size = config.get('batch_size', 32)
        self.resume_from_epoch = config.get('resume_from_epoch', 0)
        
        # Create directories
        self.checkpoint_dir = Path('checkpoints')
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Create epoch checkpoints directory
        self.epoch_checkpoint_dir = self.checkpoint_dir / 'epochs'
        self.epoch_checkpoint_dir.mkdir(exist_ok=True)
        
        # Training session ID for this run
        self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        logger.info(f"Using device: {self.device}")
        logger.info(f"Session ID: {self.session_id}")
        logger.info(f"Checkpoints will be saved to: {self.epoch_checkpoint_dir}")
    
    def save_checkpoint(self, model, optimizer, scheduler, epoch, train_loss, train_acc, 
                       val_loss, val_acc, history, is_best=False):
        """Save checkpoint after each epoch"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'history': history,
            'config': self.config,
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save epoch checkpoint
        epoch_checkpoint_path = self.epoch_checkpoint_dir / f'epoch_{epoch:03d}.pth'
        torch.save(checkpoint, epoch_checkpoint_path)
        logger.info(f"💾 Saved checkpoint: {epoch_checkpoint_path}")
        
        # Also save as latest checkpoint for easy resuming
        latest_checkpoint_path = self.checkpoint_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, latest_checkpoint_path)
        
        # Save best model separately if it's the best so far
        if is_best:
            best_model_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_model_path)
            logger.info(f"🏆 Saved best model (Val Acc: {val_acc:.2f}%)")
        
        # Save checkpoint info for tracking
        checkpoint_info_path = self.checkpoint_dir / 'checkpoint_info.json'
        checkpoint_info = {
            'latest_epoch': epoch,
            'latest_checkpoint': str(epoch_checkpoint_path),
            'session_id': self.session_id,
            'best_val_acc': val_acc if is_best else None
        }
        with open(checkpoint_info_path, 'w') as f:
            json.dump(checkpoint_info, f, indent=2)
    
    def load_checkpoint(self, model, optimizer, scheduler, checkpoint_path=None):
        """Load checkpoint to resume training"""
        if checkpoint_path is None:
            # Try to find the latest checkpoint
            latest_checkpoint = self.checkpoint_dir / 'latest_checkpoint.pth'
            if latest_checkpoint.exists():
                checkpoint_path = latest_checkpoint
            else:
                # Look for epoch checkpoints
                epoch_files = sorted(self.epoch_checkpoint_dir.glob('epoch_*.pth'))
                if epoch_files:
                    checkpoint_path = epoch_files[-1]  # Get the latest epoch
                else:
                    logger.info("No checkpoint found. Starting from scratch.")
                    return 0, {}
        
        if not Path(checkpoint_path).exists():
            logger.warning(f"Checkpoint {checkpoint_path} not found. Starting from scratch.")
            return 0, {}
        
        logger.info(f"📂 Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state if available
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Get training history
        history = checkpoint.get('history', {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        })
        
        start_epoch = checkpoint['epoch'] + 1  # Resume from next epoch
        
        logger.info(f"✅ Resumed from epoch {checkpoint['epoch']}")
        logger.info(f"   Last Train Loss: {checkpoint.get('train_loss', 'N/A'):.4f}")
        logger.info(f"   Last Train Acc: {checkpoint.get('train_acc', 'N/A'):.2f}%")
        logger.info(f"   Last Val Loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
        logger.info(f"   Last Val Acc: {checkpoint.get('val_acc', 'N/A'):.2f}%")
        
        return start_epoch, history
    
    def train_epoch(self, model, train_loader, optimizer, criterion, epoch):
        """Train for one epoch"""
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
        
        for batch_idx, (spectrograms, labels) in enumerate(pbar):
            spectrograms = spectrograms.to(self.device)
            labels = labels.to(self.device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(spectrograms)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, model, val_loader, criterion, epoch):
        """Validate model"""
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]")
            
            for spectrograms, labels in pbar:
                spectrograms = spectrograms.to(self.device)
                labels = labels.to(self.device)
                
                outputs = model(spectrograms)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                pbar.set_postfix({
                    'loss': running_loss / len(pbar),
                    'acc': 100. * correct / total
                })
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self, data_dir: str, resume=False, checkpoint_path=None):
        """Main training loop with automatic checkpointing"""
        logger.info("🎵 Starting Audio Deepfake Detection Training")
        logger.info(f"📊 Training for {self.epochs} epochs")
        logger.info(f"💾 Checkpointing enabled - saving after each epoch")
        
        # Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders(
            data_dir=data_dir,
            batch_size=self.batch_size
        )
        
        logger.info(f"Train batches: {len(train_loader)}")
        logger.info(f"Val batches: {len(val_loader)}")
        logger.info(f"Test batches: {len(test_loader)}")
        
        # Create model
        model = AudioDeepfakeDetector(
            input_size=128,
            hidden_size=256,
            num_layers=3,
            dropout=0.3,
            bidirectional=True
        )
        model = model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Initialize or load training state
        start_epoch = 0
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
        best_val_acc = 0.0
        
        # Resume from checkpoint if requested
        if resume:
            start_epoch, history = self.load_checkpoint(
                model, optimizer, scheduler, checkpoint_path
            )
            if history and 'val_acc' in history and len(history['val_acc']) > 0:
                best_val_acc = max(history['val_acc'])
                logger.info(f"Previous best validation accuracy: {best_val_acc:.2f}%")
        
        # Training loop
        logger.info(f"Starting training from epoch {start_epoch + 1}")
        
        for epoch in range(start_epoch, self.epochs):
            epoch_start_time = time.time()
            
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{self.epochs}")
            print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
            print(f"Session: {self.session_id}")
            print(f"{'='*60}")
            
            # Train
            train_loss, train_acc = self.train_epoch(
                model, train_loader, optimizer, criterion, epoch
            )
            
            # Validate
            val_loss, val_acc = self.validate(
                model, val_loader, criterion, epoch
            )
            
            # Update scheduler
            scheduler.step(val_loss)
            
            # Update history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Check if this is the best model
            is_best = val_acc > best_val_acc
            if is_best:
                best_val_acc = val_acc
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time
            
            # Print epoch summary
            print(f"\n📊 Epoch {epoch+1} Summary:")
            print(f"   Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"   Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"   Epoch Time: {epoch_time:.2f} seconds")
            if is_best:
                print(f"   🏆 New best validation accuracy!")
            
            # SAVE CHECKPOINT AFTER EACH EPOCH
            self.save_checkpoint(
                model, optimizer, scheduler, epoch,
                train_loss, train_acc, val_loss, val_acc,
                history, is_best=is_best
            )
            
            # Save training progress plot every 5 epochs
            if (epoch + 1) % 5 == 0:
                self.plot_training_progress(history, epoch + 1)
        
        # Final test evaluation
        print(f"\n{'='*60}")
        print("Final Test Evaluation")
        print(f"{'='*60}")
        
        test_loss, test_acc = self.validate(model, test_loader, criterion, -1)
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        
        # Save final results
        final_results = {
            'history': history,
            'best_val_acc': best_val_acc,
            'test_acc': test_acc,
            'total_epochs': len(history['train_loss']),
            'config': self.config,
            'session_id': self.session_id,
            'completed_at': datetime.now().isoformat()
        }
        
        results_path = self.checkpoint_dir / f'training_results_{self.session_id}.json'
        with open(results_path, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        logger.info(f"✅ Training completed! Results saved to {results_path}")
        logger.info(f"🏆 Best validation accuracy: {best_val_acc:.2f}%")
        logger.info(f"🎯 Final test accuracy: {test_acc:.2f}%")
        
        return model, history
    
    def plot_training_progress(self, history, current_epoch):
        """Save a simple text-based progress summary"""
        progress_path = self.checkpoint_dir / f'training_progress_epoch_{current_epoch:03d}.txt'
        
        with open(progress_path, 'w') as f:
            f.write(f"Training Progress - Epoch {current_epoch}\n")
            f.write(f"Session ID: {self.session_id}\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Epoch | Train Loss | Train Acc | Val Loss | Val Acc\n")
            f.write("-" * 50 + "\n")
            
            for i in range(len(history['train_loss'])):
                f.write(f"{i+1:5d} | {history['train_loss'][i]:10.4f} | "
                       f"{history['train_acc'][i]:9.2f}% | "
                       f"{history['val_loss'][i]:8.4f} | "
                       f"{history['val_acc'][i]:7.2f}%\n")
            
            f.write("\n" + "=" * 50 + "\n")
            f.write(f"Best Val Acc: {max(history['val_acc']):.2f}% "
                   f"(Epoch {history['val_acc'].index(max(history['val_acc'])) + 1})\n")

def main():
    parser = argparse.ArgumentParser(description='Train Audio Deepfake Detector with Checkpointing')
    parser.add_argument('--data-dir', type=str, default='data/raw',
                       help='Directory containing REAL and FAKE folders')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from latest checkpoint')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to specific checkpoint to resume from')
    
    args = parser.parse_args()
    
    config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate
    }
    
    print("🎵 AUDIO DEEPFAKE DETECTION TRAINING")
    print("=" * 60)
    print(f"📁 Data Directory: {args.data_dir}")
    print(f"⚙️  Configuration:")
    print(f"   - Epochs: {args.epochs}")
    print(f"   - Batch Size: {args.batch_size}")
    print(f"   - Learning Rate: {args.learning_rate}")
    print(f"   - Resume: {'Yes' if args.resume else 'No'}")
    if args.checkpoint:
        print(f"   - Checkpoint: {args.checkpoint}")
    print("=" * 60)
    
    trainer = Trainer(config)
    
    # Train with automatic checkpointing
    trainer.train(
        data_dir=args.data_dir,
        resume=args.resume,
        checkpoint_path=args.checkpoint
    )

if __name__ == "__main__":
       main()