#!/usr/bin/env python3
"""
vertex_ai_training.py - Train deepfake detector on Vertex AI with 297K images
FIXED: Added proper authentication handling
"""

import os
import sys
import argparse
import json
from pathlib import Path
from google.cloud import aiplatform
from google.cloud import storage
from google.oauth2 import service_account
from datetime import datetime
import yaml

class VertexAIDeepfakeTrainer:
    def __init__(self, project_id: str, bucket_name: str, region: str, credentials_path: str = None):
        """Initialize Vertex AI trainer with proper authentication"""
        self.project_id = project_id
        self.bucket_name = bucket_name
        self.region = region
        
        # Handle authentication
        self.credentials = None
        if credentials_path and os.path.exists(credentials_path):
            print(f"🔐 Using service account credentials: {credentials_path}")
            self.credentials = service_account.Credentials.from_service_account_file(
                credentials_path,
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
        else:
            print("🔐 Using default application credentials")
            # Let it use default credentials (gcloud auth application-default login)
        
        # Initialize Vertex AI with explicit credentials
        aiplatform.init(
            project=project_id,
            location=region,
            staging_bucket=f"gs://{bucket_name}/vertex_ai_staging",
            credentials=self.credentials
        )
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.job_name = f"deepfake-detector-{self.timestamp}"
        
        print(f"✅ Vertex AI initialized successfully")
        print(f"   Project: {project_id}")
        print(f"   Region: {region}")
        print(f"   Job Name: {self.job_name}")
    
    def test_authentication(self):
        """Test if authentication is working"""
        try:
            # Try to list something from the bucket
            client = storage.Client(credentials=self.credentials, project=self.project_id)
            bucket = client.bucket(self.bucket_name)
            
            # Just check if bucket exists
            if bucket.exists():
                print(f"✅ Authentication test passed - can access bucket: {self.bucket_name}")
                return True
            else:
                print(f"❌ Bucket {self.bucket_name} not found or not accessible")
                return False
                
        except Exception as e:
            print(f"❌ Authentication test failed: {e}")
            return False

    def create_training_config(self):
        """Create optimized training configuration for large dataset"""
        config = {
            'model': {
                'name': 'efficientnet_b4',
                'num_classes': 2,
                'pretrained': True,
                'dropout': 0.3
            },
            'training': {
                'batch_size': 32,
                'epochs': 50,
                'learning_rate': 0.0001,
                'weight_decay': 0.0001,
                'patience': 10,
                'gradient_accumulation_steps': 2,
                'mixed_precision': True,
                'num_workers': 4
            },
            'data': {
                'input_size': 224,
                'train_split': 0.7,
                'val_split': 0.2,
                'test_split': 0.1,
                'augmentation': True,
                'cache_data': False
            },
            'optimizer': {
                'name': 'AdamW',
                'betas': [0.9, 0.999],
                'eps': 1e-8
            },
            'scheduler': {
                'name': 'CosineAnnealingWarmRestarts',
                'T_0': 10,
                'T_mult': 2,
                'eta_min': 1e-6
            },
            'paths': {
                'data_dir': f"gs://{self.bucket_name}/deepfake_dataset",
                'models_dir': f"gs://{self.bucket_name}/models/{self.job_name}",
                'logs_dir': f"gs://{self.bucket_name}/logs/{self.job_name}",
                'results_dir': f"gs://{self.bucket_name}/results/{self.job_name}"
            }
        }
        
        return config
    
    def create_custom_training_script(self):
        """Create the actual training script that runs on Vertex AI"""
        script_content = '''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms as transforms
from torchvision import models
import timm
from google.cloud import storage
import json
import os
import time
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
import io
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GCSImageDataset(Dataset):
    """Dataset that loads images directly from GCS"""
    
    def __init__(self, bucket_name, categories=['real', 'fake'], transform=None, max_files_per_category=None):
        self.bucket_name = bucket_name
        self.transform = transform
        
        # Initialize GCS client
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(bucket_name)
        
        # Load file list
        self.samples = []
        for category in categories:
            prefix = f"deepfake_dataset/{category}/"
            blobs = list(self.bucket.list_blobs(prefix=prefix))
            
            label = 0 if category == 'real' else 1
            
            # Filter for image files
            image_blobs = [blob for blob in blobs if blob.name.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            # Limit files per category if specified
            if max_files_per_category:
                image_blobs = image_blobs[:max_files_per_category]
            
            for blob in image_blobs:
                self.samples.append((blob.name, label))
            
            logger.info(f"Loaded {len(image_blobs)} {category} images")
        
        logger.info(f"Total dataset size: {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        blob_name, label = self.samples[idx]
        
        try:
            # Download image from GCS
            blob = self.bucket.blob(blob_name)
            image_bytes = blob.download_as_bytes()
            
            # Load image
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            return image, label
        
        except Exception as e:
            logger.error(f"Error loading {blob_name}: {e}")
            # Return a black image as fallback
            if self.transform:
                fallback = self.transform(Image.new('RGB', (224, 224), color='black'))
            else:
                fallback = torch.zeros(3, 224, 224)
            return fallback, label

class EfficientNetDeepfakeDetector(nn.Module):
    """EfficientNet-based deepfake detector"""
    
    def __init__(self, model_name="efficientnet_b4", num_classes=2, dropout=0.3):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,
            global_pool='avg'
        )
        self.feature_dim = self.backbone.num_features
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout * 0.25),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

def train_on_vertex():
    """Main training function for Vertex AI"""
    
    # Load config from environment
    config_str = os.environ.get('TRAINING_CONFIG', '{}')
    config = json.loads(config_str)
    bucket_name = os.environ.get('BUCKET_NAME')
    job_name = os.environ.get('JOB_NAME')
    
    logger.info(f"Starting training job: {job_name}")
    logger.info(f"Using bucket: {bucket_name}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Create full dataset
    full_dataset = GCSImageDataset(bucket_name, transform=None)
    
    # Split dataset
    total_size = len(full_dataset)
    train_size = int(config['data']['train_split'] * total_size)
    val_size = int(config['data']['val_split'] * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Apply transforms
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    
    logger.info(f"Dataset splits - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )
    
    # Create model
    model = EfficientNetDeepfakeDetector(
        model_name=config['model']['name'],
        num_classes=config['model']['num_classes'],
        dropout=config['model']['dropout']
    ).to(device)
    
    # Setup training
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        betas=config['optimizer']['betas'],
        eps=config['optimizer']['eps']
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config['scheduler']['T_0'],
        T_mult=config['scheduler']['T_mult'],
        eta_min=config['scheduler']['eta_min']
    )
    
    # Mixed precision training
    scaler = GradScaler() if config['training']['mixed_precision'] else None
    
    # Training loop
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(config['training']['epochs']):
        start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']} [Train]")
        
        for batch_idx, (inputs, labels) in enumerate(train_pbar):
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            if scaler and config['training']['mixed_precision']:
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100. * train_correct / train_total:.2f}%'
            })
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']} [Val]")
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100. * val_correct / val_total:.2f}%'
                })
        
        # Calculate metrics
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        train_loss_avg = train_loss / len(train_loader)
        val_loss_avg = val_loss / len(val_loader)
        
        epoch_time = time.time() - start_time
        
        logger.info(f"Epoch {epoch+1}/{config['training']['epochs']} ({epoch_time:.1f}s):")
        logger.info(f"  Train - Loss: {train_loss_avg:.4f}, Acc: {train_acc:.2f}%")
        logger.info(f"  Val   - Loss: {val_loss_avg:.4f}, Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            # Save to local temp file first
            model_path = "/tmp/best_model.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'config': config
            }, model_path)
            
            # Upload to GCS
            try:
                client = storage.Client()
                bucket = client.bucket(bucket_name)
                blob = bucket.blob(f"models/{job_name}/best_model.pth")
                blob.upload_from_filename(model_path)
                logger.info(f"Model saved to GCS with validation accuracy: {best_val_acc:.2f}%")
            except Exception as e:
                logger.error(f"Failed to upload model to GCS: {e}")
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= config['training']['patience']:
            logger.info(f"Early stopping triggered after {epoch + 1} epochs")
            break
        
        scheduler.step()
    
    logger.info(f"Training complete! Best validation accuracy: {best_val_acc:.2f}%")
    
    # Save final training stats
    stats = {
        'best_val_acc': best_val_acc,
        'total_epochs': epoch + 1,
        'total_samples': total_size,
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'config': config
    }
    
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        stats_blob = bucket.blob(f"results/{job_name}/training_stats.json")
        stats_blob.upload_from_string(json.dumps(stats, indent=2))
        logger.info("Training stats saved to GCS")
    except Exception as e:
        logger.error(f"Failed to save stats to GCS: {e}")

if __name__ == "__main__":
    train_on_vertex()
'''
        
        # Save training script
        script_path = Path("vertex_training_script.py")
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        return script_path
    
    def create_dockerfile(self):
        """Create Dockerfile for custom container"""
        dockerfile_content = f'''
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \\
    timm==0.9.12 \\
    google-cloud-storage==2.10.0 \\
    google-cloud-aiplatform==1.38.0 \\
    pillow==10.1.0 \\
    tqdm==4.66.1 \\
    numpy==1.24.4 \\
    torchmetrics==1.2.0

# Copy training script
COPY vertex_training_script.py /app/train.py
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV BUCKET_NAME={self.bucket_name}
ENV JOB_NAME={self.job_name}

# Run training
CMD ["python", "train.py"]
'''
        
        with open("Dockerfile", 'w') as f:
            f.write(dockerfile_content)
        
        return "Dockerfile"
    
    def submit_training_job(self, machine_type: str = "n1-highmem-8", 
                           accelerator_type: str = "NVIDIA_TESLA_T4",
                           accelerator_count: int = 1):
        """Submit training job to Vertex AI"""
        
        print(f"🚀 Submitting training job: {self.job_name}")
        
        # Test authentication first
        if not self.test_authentication():
            print("❌ Authentication test failed. Please check your credentials.")
            return None
        
        # Create training config
        config = self.create_training_config()
        
        # Create training script
        script_path = self.create_custom_training_script()
        
        # Create custom container
        dockerfile_path = self.create_dockerfile()
        
        # Build and push container to Artifact Registry
        image_uri = f"gcr.io/{self.project_id}/deepfake-detector:{self.timestamp}"
        
        print("🐳 Building Docker container...")
        build_result = os.system(f"gcloud builds submit --tag {image_uri} .")
        if build_result != 0:
            print("❌ Docker build failed!")
            return None
        
        # Create custom job
        try:
            job = aiplatform.CustomJob(
                display_name=self.job_name,
                worker_pool_specs=[
                    {
                        "machine_spec": {
                            "machine_type": machine_type,
                            "accelerator_type": accelerator_type,
                            "accelerator_count": accelerator_count,
                        },
                        "replica_count": 1,
                        "container_spec": {
                            "image_uri": image_uri,
                            "env": [
                                {"name": "TRAINING_CONFIG", "value": json.dumps(config)},
                                {"name": "BUCKET_NAME", "value": self.bucket_name},
                                {"name": "JOB_NAME", "value": self.job_name},
                            ],
                        },
                    }
                ],
                staging_bucket=f"gs://{self.bucket_name}/vertex_ai_staging",
            )
            
            # Run the job
            print(f"📊 Starting training job on Vertex AI...")
            print(f"   Machine: {machine_type}")
            print(f"   GPU: {accelerator_count}x {accelerator_type}")
            print(f"   Dataset: ~297K images from GCS bucket")
            
            job.run(
                sync=False,  # Run asynchronously
                restart_job_on_worker_restart=False
            )
            
            print(f"✅ Job submitted successfully!")
            print(f"   Job Name: {job.resource_name}")
            print(f"   Monitor at: https://console.cloud.google.com/vertex-ai/training/custom-jobs")
            
            return job
            
        except Exception as e:
            print(f"❌ Failed to submit job: {e}")
            return None

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Train Deepfake Detector on Vertex AI')
    parser.add_argument('--machine-type', default='n1-highmem-8', help='Machine type')
    parser.add_argument('--gpu-type', default='NVIDIA_TESLA_T4', help='GPU type')
    parser.add_argument('--gpu-count', type=int, default=1, help='Number of GPUs')
    parser.add_argument('--project-id', required=True, help='GCP Project ID')
    parser.add_argument('--bucket-name', required=True, help='GCS Bucket name')
    parser.add_argument('--region', default='us-central1', help='GCP Region')
    parser.add_argument('--credentials-path', help='Path to service account JSON key file')
    
    args = parser.parse_args()
    
    print(f"🚀 Training Deepfake Detector on Vertex AI")
    print(f"   Project: {args.project_id}")
    print(f"   Bucket: {args.bucket_name}")
    print(f"   Region: {args.region}")
    
    # Initialize trainer with credentials
    trainer = VertexAIDeepfakeTrainer(
        args.project_id, 
        args.bucket_name, 
        args.region,
        args.credentials_path
    )
    
    # Submit training job
    job = trainer.submit_training_job(
        machine_type=args.machine_type,
        accelerator_type=args.gpu_type,
        accelerator_count=args.gpu_count
    )
    
    if job:
        print("\n🎯 Next Steps:")
        print("1. Monitor job progress in Cloud Console")
        print("2. Check logs for training metrics")
        print("3. Download trained model from GCS when complete")
        print(f"4. Model will be saved to: gs://{args.bucket_name}/models/{trainer.job_name}/")
    else:
        print("❌ Job submission failed!")

if __name__ == "__main__":
    main()