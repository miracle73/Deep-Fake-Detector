#!/usr/bin/env python3
"""
Main training script for deepfake detection model
"""

import sys
import os
from pathlib import Path
import torch
import argparse

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config.model_configs import ConfigManager
from models.efficientnet_detector import create_efficientnet_detector
from src.data.preprocessing import DeepfakePreprocessor
from src.data.dataset import create_data_loaders
from src.training.trainer import DeepfakeTrainer

def setup_device():
    """Setup training device"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        print("💻 Using CPU")
    
    return device

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train Deepfake Detection Model')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Path to data directory')
    
    args = parser.parse_args()
    
    print("🤖 DEEPFAKE DETECTION MODEL TRAINING")
    print("=" * 50)
    
    # Load configuration
    print("📝 Loading configuration...")
    config_manager = ConfigManager(args.config)
    model_config = config_manager.model_config
    training_config = config_manager.training_config
    data_config = config_manager.data_config
    path_config = config_manager.path_config
    
    print(f"   Model: {model_config.name}")
    print(f"   Epochs: {training_config.epochs}")
    print(f"   Batch Size: {training_config.batch_size}")
    print(f"   Learning Rate: {training_config.learning_rate}")
    
    # Setup device
    device = setup_device()
    
    # Create model
    print(f"\n🏗️ Creating {model_config.name} model...")
    model = create_efficientnet_detector(
        model_name=model_config.name,
        num_classes=model_config.num_classes,
        pretrained=model_config.pretrained,
        dropout=model_config.dropout
    )
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Create preprocessor
    print(f"\n📸 Setting up data preprocessing...")
    preprocessor = DeepfakePreprocessor(
        input_size=data_config.input_size,
        normalize=True,
        augment=data_config.augmentation
    )
    
    # Create data loaders
    print(f"📊 Loading dataset from {args.data_dir}...")
    try:
        train_loader, val_loader, test_loader = create_data_loaders(
            data_dir=args.data_dir,
            preprocessor=preprocessor,
            batch_size=training_config.batch_size,
            num_workers=4,
            balance_classes=True
        )
        
        print(f"   Training samples: {len(train_loader.dataset)}")
        print(f"   Validation samples: {len(val_loader.dataset)}")
        print(f"   Test samples: {len(test_loader.dataset)}")
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print("💡 Make sure to run the data download script first!")
        return
    
    # Create trainer
    print(f"\n🎓 Setting up trainer...")
    trainer = DeepfakeTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config.__dict__,
        device=device,
        checkpoint_dir=path_config.models_dir,
        log_dir=path_config.logs_dir
    )
    
    # Start training
    print(f"\n🚀 Starting training...")
    try:
        history = trainer.train(resume_from_checkpoint=args.resume)
        
        print(f"\n✅ Training completed successfully!")
        print(f"📊 Final Results:")
        print(f"   Best Validation Accuracy: {trainer.best_val_acc:.4f}")
        print(f"   Final Training Accuracy: {history['train_acc'][-1]:.4f}")
        
        # Save training history
        import json
        history_path = Path(path_config.results_dir) / "training_history.json"
        history_path.parent.mkdir(exist_ok=True)
        
        # Convert numpy values to Python types for JSON serialization
        json_history = {}
        for key, values in history.items():
            json_history[key] = [float(v) for v in values]
        
        with open(history_path, 'w') as f:
            json.dump(json_history, f, indent=2)
        
        print(f"📈 Training history saved to {history_path}")
        print(f"💾 Best model saved to {Path(path_config.models_dir) / 'best_model.pth'}")
        
    except KeyboardInterrupt:
        print(f"\n⏹️ Training interrupted by user")
        print(f"💾 Latest checkpoint saved to {Path(path_config.models_dir) / 'latest_checkpoint.pth'}")
    
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        raise

if __name__ == "__main__":
    main()