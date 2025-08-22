#!/usr/bin/env python3
"""
Train Audio Deepfake Detection Model
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import json
import time

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
    """Audio deepfake detection trainer"""
    
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Training parameters
        self.epochs = config.get('epochs', 50)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.batch_size = config.get('batch_size', 32)
        
        # Create directories
        self.checkpoint_dir = Path('checkpoints')
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        logger.info(f"Using device: {self.device}")
    
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
    
    def train(self, data_dir: str):
        """Main training loop"""
        logger.info("🎵 Starting Audio Deepfake Detection Training")
        
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
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Training history
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
        
        best_val_acc = 0.0
        
        # Training loop
        for epoch in range(self.epochs):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch+1}/{self.epochs}")
            print(f"{'='*50}")
            
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
            
            # Save history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Print epoch summary
            print(f"\nTrain Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'config': self.config
                }, self.checkpoint_dir / 'best_model.pth')
                print(f"✅ Saved best model (Val Acc: {val_acc:.2f}%)")
        
        # Final test evaluation
        print(f"\n{'='*50}")
        print("Final Test Evaluation")
        print(f"{'='*50}")
        
        test_loss, test_acc = self.validate(model, test_loader, criterion, -1)
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        
        # Save final results
        with open('training_results.json', 'w') as f:
            json.dump({
                'history': history,
                'best_val_acc': best_val_acc,
                'test_acc': test_acc,
                'config': self.config
            }, f, indent=2)
        
        logger.info("✅ Training completed!")
        return model, history

def main():
    parser = argparse.ArgumentParser(description='Train Audio Deepfake Detector')
    parser.add_argument('--data-dir', type=str, default='data/raw',
                       help='Directory containing REAL and FAKE folders')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate')
    
    args = parser.parse_args()
    
    config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate
    }
    
    trainer = Trainer(config)
    trainer.train(args.data_dir)

if __name__ == "__main__":
    main()