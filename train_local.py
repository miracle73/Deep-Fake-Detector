#!/usr/bin/env python3
"""
Local training script for GitHub Codespaces
"""

import os
import sys
import torch
import logging
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from scripts.vertex_ai_training_cpu_optimized import CPUOptimizedTrainer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Configuration optimized for Codespace
    config = {
        'batch_size': 2,  # Even smaller for Codespace memory limits
        'epochs': 5,       # Fewer epochs for quick testing
        'learning_rate': 0.001,
        'max_frames': 16,
        'frame_encoder': 'mobilenet_v2',
        'temporal_model': 'lstm',
        'gradient_accumulation_steps': 8,  # Compensate for small batch
        'weight_decay': 1e-4
    }
    
    # Your GCS bucket
    bucket_name = 'second-bucket-video-deepfake'
    
    print("🎬 LOCAL VIDEO DEEPFAKE TRAINING")
    print("=" * 50)
    print(f"🌐 Environment: GitHub Codespaces")
    print(f"☁️ Data Source: gs://{bucket_name}/processed/")
    print(f"📊 Batch Size: {config['batch_size']}")
    print(f"⚡ Epochs: {config['epochs']}")
    print("=" * 50)
    
    try:
        # Initialize trainer
        trainer = CPUOptimizedTrainer(config)
        
        # Train model
        print("🚀 Starting training...")
        model, results = trainer.train(bucket_name)
        
        # Save model locally
        output_dir = "./trained_models"
        model_path, results_path = trainer.save_model(model, results, output_dir)
        
        print(f"\n✅ Training completed!")
        print(f"📊 Best Validation Accuracy: {results['best_val_accuracy']:.4f}")
        print(f"💾 Model saved to: {model_path}")
        
        # Optionally upload to GCS
        upload = input("\nUpload model to GCS? (y/n): ")
        if upload.lower() == 'y':
            from google.cloud import storage
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            
            blob = bucket.blob(f"models/local_training/model.pth")
            blob.upload_from_filename(model_path)
            print(f"☁️ Model uploaded to: gs://{bucket_name}/models/local_training/")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise

if __name__ == "__main__":
    main()