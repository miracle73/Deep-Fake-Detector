#!/usr/bin/env python3
"""
vertex_ai_training_cpu_optimized.py - CPU-optimized training for deepfake detection
This version is optimized for CPU training with a sampled subset of the data
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
import time

# CPU-optimized training code with data sampling
TRAINING_CODE = '''
import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from PIL import Image
from google.cloud import storage
import logging
from io import BytesIO
import random
from tqdm import tqdm
import gc

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class SampledGCSDataset(Dataset):
    """Sampled dataset for CPU training - loads a subset of images"""
    
    def __init__(self, bucket_name, prefix="deepfake_dataset", 
                 sample_size=10000, balanced=True):
        """
        Args:
            bucket_name: GCS bucket name
            prefix: Dataset prefix in bucket
            sample_size: Number of images to sample (total)
            balanced: If True, sample equally from real/fake
        """
        self.bucket_name = bucket_name
        self.samples = []
        
        logger.info(f"Initializing sampled dataset from gs://{bucket_name}/{prefix}")
        logger.info(f"Target sample size: {sample_size} images")
        
        # Initialize GCS client
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        
        # Collect all available files
        real_files = []
        fake_files = []
        
        # List real images
        real_prefix = f"{prefix}/real/"
        logger.info(f"Scanning real images...")
        for blob in bucket.list_blobs(prefix=real_prefix):
            if blob.name.lower().endswith((".jpg", ".jpeg", ".png")):
                real_files.append(blob.name)
        
        # List fake images
        fake_prefix = f"{prefix}/fake/"
        logger.info(f"Scanning fake images...")
        for blob in bucket.list_blobs(prefix=fake_prefix):
            if blob.name.lower().endswith((".jpg", ".jpeg", ".png")):
                fake_files.append(blob.name)
        
        logger.info(f"Found {len(real_files)} real and {len(fake_files)} fake images")
        
        # Sample from the files
        if balanced:
            # Sample equally from both classes
            samples_per_class = sample_size // 2
            
            # Sample real images
            if len(real_files) > samples_per_class:
                sampled_real = random.sample(real_files, samples_per_class)
            else:
                sampled_real = real_files
                logger.warning(f"Only {len(real_files)} real images available")
            
            # Sample fake images
            if len(fake_files) > samples_per_class:
                sampled_fake = random.sample(fake_files, samples_per_class)
            else:
                sampled_fake = fake_files
                logger.warning(f"Only {len(fake_files)} fake images available")
            
            # Add to samples list
            for blob_name in sampled_real:
                self.samples.append((blob_name, 0))  # 0 for real
            for blob_name in sampled_fake:
                self.samples.append((blob_name, 1))  # 1 for fake
        else:
            # Random sampling from all files
            all_files = [(f, 0) for f in real_files] + [(f, 1) for f in fake_files]
            if len(all_files) > sample_size:
                self.samples = random.sample(all_files, sample_size)
            else:
                self.samples = all_files
        
        # Shuffle samples
        random.shuffle(self.samples)
        
        logger.info(f"Dataset initialized with {len(self.samples)} samples")
        real_count = sum(1 for _, label in self.samples if label == 0)
        fake_count = len(self.samples) - real_count
        logger.info(f"Class distribution: {real_count} real, {fake_count} fake")
        
        # Define transforms - simpler for CPU
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Fixed size, no random crops
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Initialize client for loading
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        blob_name, label = self.samples[idx]
        
        try:
            # Download image from GCS to memory
            blob = self.bucket.blob(blob_name)
            img_bytes = blob.download_as_bytes()
            
            # Open image
            image = Image.open(BytesIO(img_bytes)).convert("RGB")
            
            # Apply transforms
            if self.transform:
                image = self.transform(image)
            
            return image, label
            
        except Exception as e:
            logger.warning(f"Failed to load {blob_name}: {e}")
            # Return a black image as fallback
            return torch.zeros(3, 224, 224), label

class MobileNetDetector(nn.Module):
    """MobileNetV2 - more efficient for CPU than EfficientNet"""
    
    def __init__(self, num_classes=2, dropout=0.2):
        super().__init__()
        
        logger.info("Initializing MobileNetV2 model (optimized for CPU)")
        
        # Use MobileNetV2 - much faster on CPU than EfficientNet
        self.base_model = models.mobilenet_v2(pretrained=True)
        
        # Get the number of input features for the classifier
        num_features = self.base_model.classifier[1].in_features
        
        # Replace classifier for binary classification
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(num_features, num_classes)
        )
    
    def forward(self, x):
        return self.base_model(x)

def train_epoch(model, dataloader, criterion, optimizer, device, epoch, total_epochs):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{total_epochs}")
    for batch_idx, (inputs, labels) in enumerate(pbar):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        accuracy = 100. * correct / total
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{accuracy:.2f}%"})
        
        # Log every 50 batches
        if batch_idx % 50 == 0 and batch_idx > 0:
            logger.info(f"Epoch {epoch} [{batch_idx}/{len(dataloader)}] "
                       f"Loss: {running_loss/batch_idx:.4f} Acc: {accuracy:.2f}%")
        
        # Clear cache periodically on CPU
        if batch_idx % 100 == 0:
            gc.collect()
    
    return running_loss / len(dataloader), 100. * correct / total

def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validation"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return val_loss / len(dataloader), 100. * correct / total

def main():
    logger.info("=" * 60)
    logger.info("DEEPFAKE DETECTOR - CPU-OPTIMIZED TRAINING")
    logger.info("=" * 60)
    
    # Get environment variables
    bucket_name = os.environ.get("BUCKET_NAME", "")
    job_name = os.environ.get("JOB_NAME", "deepfake_training")
    sample_size = int(os.environ.get("SAMPLE_SIZE", "10000"))
    
    if not bucket_name:
        logger.error("No BUCKET_NAME specified!")
        sys.exit(1)
    
    # Force CPU usage
    device = torch.device("cpu")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"Using device: CPU (optimized for CPU training)")
    
    # CPU-optimized configuration
    config = {
        "sample_size": sample_size,  # Use subset of data
        "batch_size": 8,  # Small batch size for CPU
        "epochs": 20,  # More epochs since we have less data
        "learning_rate": 0.001,
        "num_workers": 2,  # Limited workers for CPU
        "validation_split": 0.2,
    }
    
    logger.info(f"Configuration: {json.dumps(config, indent=2)}")
    logger.info(f"Training on {config['sample_size']} sampled images (from ~297K total)")
    
    # Create sampled dataset
    logger.info(f"Loading sampled dataset from gs://{bucket_name}/deepfake_dataset")
    dataset = SampledGCSDataset(
        bucket_name=bucket_name,
        prefix="deepfake_dataset",
        sample_size=config["sample_size"],
        balanced=True  # Ensure balanced classes
    )
    
    if len(dataset) == 0:
        logger.error("No images found in dataset!")
        sys.exit(1)
    
    # Split dataset
    train_size = int((1 - config["validation_split"]) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    logger.info(f"Dataset split: {train_size} train, {val_size} validation")
    
    # Create dataloaders with CPU optimization
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=False,  # No GPU
        persistent_workers=True if config["num_workers"] > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=False,
        persistent_workers=True if config["num_workers"] > 0 else False
    )
    
    # Create model - MobileNetV2 for CPU
    model = MobileNetDetector(num_classes=2, dropout=0.2).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )
    
    # Training loop
    logger.info("Starting CPU-optimized training...")
    logger.info(f"Estimated time: {config['epochs'] * train_size // config['batch_size'] // 60} minutes")
    
    best_acc = 0
    patience_counter = 0
    max_patience = 5
    
    for epoch in range(1, config["epochs"] + 1):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, config["epochs"]
        )
        
        # Validation
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        logger.info(f"Epoch {epoch}/{config['epochs']}: "
                   f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                   f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Update learning rate based on validation accuracy
        scheduler.step(val_acc)
        
        # Early stopping
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            
            # Save best model
            logger.info(f"New best model! Val Acc: {val_acc:.2f}%")
            
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_acc": train_acc,
                "val_acc": val_acc,
                "config": config,
                "sample_size": config["sample_size"]
            }
            
            local_path = "/tmp/best_model.pth"
            torch.save(checkpoint, local_path)
            
            # Upload to GCS
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(f"models/{job_name}/best_model_cpu_{config['sample_size']}_samples.pth")
            blob.upload_from_filename(local_path)
            
            logger.info(f"Model saved to gs://{bucket_name}/models/{job_name}/")
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                logger.info(f"Early stopping triggered after {epoch} epochs")
                break
        
        # Clear cache
        gc.collect()
    
    logger.info(f"Training completed! Best validation accuracy: {best_acc:.2f}%")
    logger.info(f"Trained on {config['sample_size']} samples from ~297K total images")
    
    # Save training summary
    summary = {
        "total_images_available": "~297K",
        "images_used_for_training": config["sample_size"],
        "best_validation_accuracy": best_acc,
        "model_type": "MobileNetV2",
        "training_device": "CPU",
        "config": config
    }
    
    summary_path = "/tmp/training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(f"models/{job_name}/training_summary.json")
    blob.upload_from_filename(summary_path)
    
    logger.info("Training summary saved to GCS")
    logger.info("=" * 60)
    logger.info("NEXT STEPS:")
    logger.info("1. This model was trained on a sample for CPU efficiency")
    logger.info("2. For production, request GPU quota and train on full dataset")
    logger.info("3. Model can be used for initial testing and validation")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
'''

class VertexAIDeepfakeTrainer:
    def __init__(self, project_id: str, bucket_name: str, region: str, credentials_path: str = None):
        """Initialize Vertex AI trainer with proper authentication"""
        self.project_id = project_id
        self.bucket_name = bucket_name
        self.region = region
        
        # Handle authentication
        self.credentials = None
        credentials_file = None
        
        if credentials_path and os.path.exists(credentials_path):
            credentials_file = credentials_path
            print(f"🔐 Using explicit credentials: {credentials_path}")
        elif os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
            credentials_file = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
            if os.path.exists(credentials_file):
                print(f"🔐 Using GOOGLE_APPLICATION_CREDENTIALS: {credentials_file}")
        else:
            print("🔐 Attempting to use default credentials")
        
        if credentials_file:
            try:
                self.credentials = service_account.Credentials.from_service_account_file(
                    credentials_file,
                    scopes=["https://www.googleapis.com/auth/cloud-platform"]
                )
                print(f"✅ Successfully loaded service account credentials")
            except Exception as e:
                print(f"❌ Failed to load credentials: {e}")
                self.credentials = None
        
        # Initialize Vertex AI
        aiplatform.init(
            project=project_id,
            location=region,
            staging_bucket=f"gs://{bucket_name}/vertex_ai_staging",
            credentials=self.credentials
        )
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.job_name = f"deepfake-cpu-optimized-{self.timestamp}"
        
        print(f"✅ Vertex AI initialized")
    
    def test_authentication(self):
        """Test if authentication is working"""
        try:
            client = storage.Client(credentials=self.credentials, project=self.project_id)
            bucket = client.bucket(self.bucket_name)
            
            if bucket.exists():
                print(f"✅ Can access bucket: {self.bucket_name}")
                
                # Count approximate number of images
                real_count = 0
                fake_count = 0
                
                print("📊 Checking dataset size...")
                for blob in bucket.list_blobs(prefix="deepfake_dataset/real/", max_results=1000):
                    if blob.name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        real_count += 1
                
                for blob in bucket.list_blobs(prefix="deepfake_dataset/fake/", max_results=1000):
                    if blob.name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        fake_count += 1
                
                print(f"✅ Found dataset with {real_count}+ real and {fake_count}+ fake images")
                return True
            else:
                print(f"❌ Bucket {self.bucket_name} not found")
                return False
                
        except Exception as e:
            print(f"❌ Authentication test failed: {e}")
            return False
    
    def submit_training_job(self, machine_type: str = "n1-highmem-8", 
                           sample_size: int = 10000):
        """Submit CPU-optimized training job"""
        
        print(f"\n🚀 Submitting CPU-optimized training job: {self.job_name}")
        print(f"📊 Configuration:")
        print(f"   - Sample size: {sample_size} images (from ~297K)")
        print(f"   - Machine type: {machine_type}")
        print(f"   - Model: MobileNetV2 (CPU-optimized)")
        print(f"   - Strategy: Balanced sampling from both classes")
        
        if not self.test_authentication():
            print("❌ Authentication failed")
            return None
        
        # Use existing Docker image
        image_uri = f"gcr.io/{self.project_id}/deepfake-detector:20250809_175159"
        print(f"♻️  Using existing Docker image: {image_uri}")
        
        # CPU-only machine spec
        machine_spec = {
            "machine_type": machine_type,
        }
        
        # Worker pool configuration
        worker_pool_specs = [
            {
                "machine_spec": machine_spec,
                "replica_count": 1,
                "container_spec": {
                    "image_uri": image_uri,
                    "command": ["python", "-c", TRAINING_CODE],
                    "env": [
                        {"name": "BUCKET_NAME", "value": self.bucket_name},
                        {"name": "JOB_NAME", "value": self.job_name},
                        {"name": "SAMPLE_SIZE", "value": str(sample_size)},
                        {"name": "PYTHONUNBUFFERED", "value": "1"},
                    ],
                },
            }
        ]
        
        try:
            # Create and submit job
            job = aiplatform.CustomJob(
                display_name=self.job_name,
                worker_pool_specs=worker_pool_specs,
                staging_bucket=f"gs://{self.bucket_name}/vertex_ai_staging",
            )
            
            print(f"✅ Submitting job...")
            job.submit()
            
            time.sleep(5)
            
            print(f"\n✅ Job submitted successfully!")
            print(f"   Job Name: {job.display_name}")
            print(f"\n📊 Monitor at:")
            print(f"   https://console.cloud.google.com/vertex-ai/training/custom-jobs?project={self.project_id}")
            print(f"\n📁 Model will be saved to:")
            print(f"   gs://{self.bucket_name}/models/{self.job_name}/")
            
            # Estimate training time
            estimated_time = (sample_size // 1000) * 3  # ~3 min per 1000 images on CPU
            print(f"\n⏰ Estimated training time: {estimated_time}-{estimated_time*2} minutes")
            print(f"   (Training on {sample_size} sampled images)")
            
            return job
            
        except Exception as e:
            print(f"❌ Failed to submit job: {e}")
            return None

def main():
    parser = argparse.ArgumentParser(description='CPU-Optimized Deepfake Training')
    parser.add_argument('--machine-type', default='n1-highmem-8', help='Machine type')
    parser.add_argument('--sample-size', type=int, default=10000, 
                       help='Number of images to sample for training (default: 10000)')
    parser.add_argument('--project-id', required=True, help='GCP Project ID')
    parser.add_argument('--bucket-name', required=True, help='GCS Bucket name')
    parser.add_argument('--region', default='us-central1', help='GCP Region')
    parser.add_argument('--credentials-path', help='Path to service account JSON')
    
    args = parser.parse_args()
    
    print(f"🎯 CPU-OPTIMIZED DEEPFAKE DETECTION TRAINING")
    print(f"=" * 60)
    print(f"📋 Strategy:")
    print(f"   1. Sample {args.sample_size} images from ~297K total")
    print(f"   2. Use MobileNetV2 (faster on CPU than EfficientNet)")
    print(f"   3. Train with small batches to avoid memory issues")
    print(f"   4. Save best model based on validation accuracy")
    print(f"=" * 60)
    
    # Initialize trainer
    trainer = VertexAIDeepfakeTrainer(
        args.project_id, 
        args.bucket_name, 
        args.region,
        args.credentials_path
    )
    
    # Submit job
    job = trainer.submit_training_job(
        machine_type=args.machine_type,
        sample_size=args.sample_size
    )
    
    if job:
        print("\n✅ SUCCESS! CPU-optimized training started")
        print("\n📝 What this does:")
        print(f"   - Trains on {args.sample_size} sampled images")
        print("   - Creates a baseline model for testing")
        print("   - Validates your pipeline works end-to-end")
        print("\n🎯 For production:")
        print("   1. Request GPU quota from GCP")
        print("   2. Train on full 297K dataset with GPU")
        print("   3. Use EfficientNet-B4 for better accuracy")
    else:
        print("\n❌ Job submission failed!")

if __name__ == "__main__":
    main()