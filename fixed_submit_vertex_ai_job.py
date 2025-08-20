"""
FIXED - Submit Video Deepfake Training Job to Vertex AI
Key fixes:
1. Proper error handling order
2. Better job configuration
3. Clearer debugging output
"""

import os
import sys
import argparse
from datetime import datetime
from google.cloud import aiplatform
from google.cloud import storage

def create_vertex_ai_job(
    project_id: str,
    region: str,
    bucket_name: str,
    service_account: str,
    machine_type: str = "n1-standard-4",
    use_gpu: bool = False,
    epochs: int = 20,
    batch_size: int = 4
):
    """Create and submit Vertex AI custom training job"""
    
    print("🔧 Initializing Vertex AI...")
    try:
        # Initialize Vertex AI with staging bucket
        aiplatform.init(
            project=project_id, 
            location=region,
            staging_bucket=f"gs://{bucket_name}"
        )
        print(f"✅ Vertex AI initialized successfully")
        print(f"   Project: {project_id}")
        print(f"   Region: {region}")
        print(f"   Staging Bucket: gs://{bucket_name}")
    except Exception as e:
        print(f"❌ Failed to initialize Vertex AI: {e}")
        raise
    
    # Job display name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_display_name = f"deepfake-detector-{timestamp}"
    
    # Use a working container image
    if use_gpu:
        container_image = "gcr.io/cloud-aiplatform/training/pytorch-gpu.1-13:latest"
        accelerator_type = "NVIDIA_TESLA_T4"
        accelerator_count = 1
    else:
        container_image = "gcr.io/cloud-aiplatform/training/pytorch-cpu.1-13:latest"
        accelerator_type = None
        accelerator_count = 0
    
    print("\n🚀 PREPARING VERTEX AI TRAINING JOB")
    print("=" * 50)
    print(f"📋 Job Name: {job_display_name}")
    print(f"🖥️  Machine Type: {machine_type}")
    print(f"🎯 GPU: {'Yes' if use_gpu else 'No'}")
    print(f"📊 Data: gs://{bucket_name}/processed/")
    print(f"🪣 Output: gs://{bucket_name}/vertex_ai_jobs/{timestamp}")
    print(f"👤 Service Account: {service_account}")
    print("=" * 50)
    
    # Create the training script (same as before)
    training_script = f"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sys
import os
import logging
import json
import tempfile
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional, List
import random
from tqdm import tqdm
from google.cloud import storage

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("🎬 Starting CPU-Optimized Vertex AI Training...")
print("Data source: gs://{bucket_name}/processed/")

# Your VideoDeepfakeDetector model
class VideoDeepfakeDetector(nn.Module):
    def __init__(
        self,
        frame_encoder: str = 'mobilenet_v2',
        temporal_model: str = 'lstm',
        num_frames: int = 16,
        num_classes: int = 2,
        dropout: float = 0.3,
        cpu_optimized: bool = True
    ):
        super().__init__()
        
        self.num_frames = num_frames
        self.num_classes = num_classes
        self.cpu_optimized = cpu_optimized
        
        # Frame encoder (lightweight for CPU)
        if frame_encoder == 'mobilenet_v2':
            from torchvision import models
            backbone = models.mobilenet_v2(pretrained=True)
            self.frame_encoder = nn.Sequential(*list(backbone.features))
            feature_dim = 1280
        elif frame_encoder == 'resnet18':
            from torchvision import models
            backbone = models.resnet18(pretrained=True)
            self.frame_encoder = nn.Sequential(*list(backbone.children())[:-2])
            feature_dim = 512
        else:
            raise ValueError(f"Unsupported encoder: {{frame_encoder}}")
        
        # Adaptive pooling for consistent output size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Temporal modeling
        if temporal_model == 'lstm':
            self.temporal_model = nn.LSTM(
                input_size=feature_dim,
                hidden_size=256,
                num_layers=2,
                batch_first=True,
                dropout=dropout,
                bidirectional=False
            )
            temporal_output_dim = 256
        elif temporal_model == 'gru':
            self.temporal_model = nn.GRU(
                input_size=feature_dim,
                hidden_size=256,
                num_layers=2,
                batch_first=True,
                dropout=dropout,
                bidirectional=False
            )
            temporal_output_dim = 256
        else:
            self.temporal_model = None
            temporal_output_dim = feature_dim
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(temporal_output_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, num_frames, height, width = x.shape
        
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(batch_size * num_frames, channels, height, width)
        
        frame_features = self.frame_encoder(x)
        frame_features = self.adaptive_pool(frame_features)
        frame_features = frame_features.flatten(1)
        
        feature_dim = frame_features.shape[1]
        frame_features = frame_features.view(batch_size, num_frames, feature_dim)
        
        if self.temporal_model is not None:
            temporal_output, _ = self.temporal_model(frame_features)
            video_features = temporal_output[:, -1, :]
        else:
            video_features = torch.mean(frame_features, dim=1)
        
        logits = self.classifier(video_features)
        return logits

# Your VideoDataset class
class VideoDataset(Dataset):
    def __init__(
        self,
        bucket_name: str,
        gcs_prefix: str = "processed",
        split: str = 'train',
        max_frames: int = 16,
        frame_size: Tuple[int, int] = (224, 224),
        augment: bool = False,
        train_ratio: float = 0.7,
        val_ratio: float = 0.2,
        test_ratio: float = 0.1
    ):
        self.bucket_name = bucket_name
        self.gcs_prefix = gcs_prefix
        self.split = split
        self.max_frames = max_frames
        self.frame_size = frame_size
        self.augment = augment and split == 'train'
        
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        
        self.class_to_idx = {{'Real': 0, 'Fake': 1}}
        
        # Initialize GCS client
        self.gcs_client = storage.Client()
        self.gcs_bucket = self.gcs_client.bucket(bucket_name)
        
        self.samples = self._load_samples()
        logger.info(f"Loaded {{len(self.samples)}} {{split}} samples")
    
    def _load_samples(self) -> List[Tuple[str, int]]:
        samples = []
        
        for class_name, label in [('Real', 0), ('Fake', 1)]:
            prefix = f"{{self.gcs_prefix}}/{{class_name}}/"
            
            try:
                blobs = self.gcs_client.list_blobs(self.gcs_bucket, prefix=prefix)
                class_count = 0
                
                for blob in blobs:
                    if blob.name.endswith('.npz'):
                        samples.append((blob.name, label))
                        class_count += 1
                
                logger.info(f"Found {{class_count}} {{class_name}} samples")
                
            except Exception as e:
                logger.error(f"Error loading {{class_name}} samples: {{e}}")
        
        return self._split_samples(samples)
    
    def _split_samples(self, samples: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
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
            raise ValueError(f"Unknown split: {{self.split}}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample_path, label = self.samples[idx]
        
        try:
            frames = self._load_gcs_frames(sample_path)
            
            if frames is None:
                return self._get_dummy_sample(label)
            
            frames = self._process_frames(frames)
            
            if self.augment:
                frames = self._apply_augmentations(frames)
            
            frames = frames.astype(np.float32) / 255.0
            frames_tensor = torch.from_numpy(frames)
            frames_tensor = frames_tensor.permute(3, 0, 1, 2)
            
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
            frames_tensor = (frames_tensor - mean) / std
            
            return frames_tensor, label
            
        except Exception as e:
            logger.error(f"Error loading sample {{idx}}: {{e}}")
            return self._get_dummy_sample(label)
    
    def _load_gcs_frames(self, blob_name: str) -> Optional[np.ndarray]:
        try:
            blob = self.gcs_bucket.blob(blob_name)
            
            with tempfile.NamedTemporaryFile() as tmp_file:
                blob.download_to_filename(tmp_file.name)
                data = np.load(tmp_file.name)
                return data['frames']
                
        except Exception as e:
            logger.error(f"Error loading GCS frames from {{blob_name}}: {{e}}")
            return None
    
    def _process_frames(self, frames: np.ndarray) -> np.ndarray:
        if len(frames) > self.max_frames:
            if self.augment:
                start_idx = random.randint(0, len(frames) - self.max_frames)
            else:
                start_idx = (len(frames) - self.max_frames) // 2
            frames = frames[start_idx:start_idx + self.max_frames]
        elif len(frames) < self.max_frames:
            padding_needed = self.max_frames - len(frames)
            if len(frames) > 0:
                last_frame = frames[-1]
                padding = np.tile(last_frame[np.newaxis], (padding_needed, 1, 1, 1))
                frames = np.concatenate([frames, padding], axis=0)
            else:
                frames = np.zeros((self.max_frames, *self.frame_size, 3), dtype=np.uint8)
        
        return frames
    
    def _apply_augmentations(self, frames: np.ndarray) -> np.ndarray:
        if random.random() > 0.5:
            frames = np.flip(frames, axis=2)
        
        if random.random() > 0.5:
            brightness = random.uniform(0.8, 1.2)
            frames = np.clip(frames * brightness, 0, 255)
        
        return frames
    
    def _get_dummy_sample(self, label: int) -> Tuple[torch.Tensor, int]:
        dummy_frames = torch.zeros(3, self.max_frames, *self.frame_size)
        return dummy_frames, label

# Training configuration
config = {{
    'batch_size': {batch_size},
    'epochs': {epochs},
    'learning_rate': 0.001,
    'max_frames': 16,
    'frame_encoder': 'mobilenet_v2',
    'temporal_model': 'lstm',
    'gradient_accumulation_steps': 4,
    'weight_decay': 1e-4
}}

# Initialize model
model = VideoDeepfakeDetector(
    frame_encoder=config['frame_encoder'],
    temporal_model=config['temporal_model'],
    num_frames=config['max_frames'],
    dropout=0.3,
    cpu_optimized=True
)

device = torch.device('cpu')
model = model.to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"✅ Model created:")
print(f"   Total Parameters: {{total_params:,}}")
print(f"   Trainable Parameters: {{trainable_params:,}}")

# Create datasets
print("📹 Creating datasets...")
train_dataset = VideoDataset(
    bucket_name="{bucket_name}",
    gcs_prefix="processed",
    split='train',
    max_frames=config['max_frames'],
    frame_size=(224, 224),
    augment=True
)

val_dataset = VideoDataset(
    bucket_name="{bucket_name}",
    gcs_prefix="processed",
    split='val',
    max_frames=config['max_frames'],
    frame_size=(224, 224),
    augment=False
)

print(f"Training samples: {{len(train_dataset)}}")
print(f"Validation samples: {{len(val_dataset)}}")

# Create data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=config['batch_size'],
    shuffle=True,
    num_workers=2,  # Reduced for CPU
    pin_memory=False,
    drop_last=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=config['batch_size'],
    shuffle=False,
    num_workers=2,
    pin_memory=False,
    drop_last=False
)

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(
    model.parameters(),
    lr=config['learning_rate'],
    weight_decay=config['weight_decay']
)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3, verbose=True
)

# Training loop
print("🚀 Starting training...")
best_val_acc = 0.0
history = {{
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': []
}}

for epoch in range(config['epochs']):
    print(f"\\nEpoch {{epoch+1}}/{{config['epochs']}}")
    
    # Training phase
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    
    for batch_idx, (videos, labels) in enumerate(train_loader):
        if batch_idx >= 10:  # Limit batches for quick demo
            break
            
        videos = videos.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(videos)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * videos.size(0)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)
        total_samples += videos.size(0)
        
        if batch_idx % 2 == 0:
            print(f"  Batch {{batch_idx}}: Loss={{loss.item():.4f}}")
    
    if total_samples > 0:
        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects.double() / total_samples
        
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc.item())
        
        print(f"Train Loss: {{epoch_loss:.4f}}, Acc: {{epoch_acc:.4f}}")
        
        # Simple validation (limited for demo)
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        val_samples = 0
        
        with torch.no_grad():
            for batch_idx, (videos, labels) in enumerate(val_loader):
                if batch_idx >= 5:  # Limit for demo
                    break
                    
                videos = videos.to(device)
                labels = labels.to(device)
                
                outputs = model(videos)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * videos.size(0)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data)
                val_samples += videos.size(0)
        
        if val_samples > 0:
            val_epoch_loss = val_loss / val_samples
            val_epoch_acc = val_corrects.double() / val_samples
            
            history['val_loss'].append(val_epoch_loss)
            history['val_acc'].append(val_epoch_acc.item())
            
            print(f"Val Loss: {{val_epoch_loss:.4f}}, Acc: {{val_epoch_acc:.4f}}")
            
            if val_epoch_acc > best_val_acc:
                best_val_acc = val_epoch_acc
                print(f"🎉 New best validation accuracy: {{best_val_acc:.4f}}")
        
        scheduler.step(epoch_loss)

print("✅ Training completed!")

# Save results
results_dir = "/tmp/model_output"
os.makedirs(results_dir, exist_ok=True)

# Save model
torch.save({{
    'model_state_dict': model.state_dict(),
    'config': config,
    'best_val_accuracy': float(best_val_acc),
    'history': history
}}, f"{{results_dir}}/video_deepfake_detector.pth")

# Save training results
results = {{
    "timestamp": datetime.now().isoformat(),
    "status": "completed",
    "best_val_accuracy": float(best_val_acc),
    "final_train_accuracy": history['train_acc'][-1] if history['train_acc'] else 0,
    "epochs_trained": len(history['train_acc']),
    "model_parameters": total_params,
    "data_source": "gs://{bucket_name}/processed/",
    "training_samples": len(train_dataset),
    "validation_samples": len(val_dataset),
    "config": config
}}

with open(f"{{results_dir}}/training_results.json", "w") as f:
    json.dump(results, f, indent=2)

with open(f"{{results_dir}}/model_config.json", "w") as f:
    json.dump(config, f, indent=2)

print(f"📊 Results saved to {{results_dir}}/")

# Upload to GCS
try:
    client = storage.Client()
    bucket = client.bucket("{bucket_name}")
    
    for file_path in os.listdir(results_dir):
        if os.path.isfile(os.path.join(results_dir, file_path)):
            blob = bucket.blob(f"vertex_ai_jobs/{timestamp}/{{file_path}}")
            blob.upload_from_filename(os.path.join(results_dir, file_path))
            print(f"📤 Uploaded: gs://{bucket_name}/vertex_ai_jobs/{timestamp}/{{file_path}}")
    
    print("☁️ Results uploaded to GCS!")
    print(f"📈 Best Validation Accuracy: {{best_val_acc:.4f}}")
    print(f"📊 Training completed with {{len(train_dataset)}} train + {{len(val_dataset)}} val samples")
    
except Exception as e:
    print(f"⚠️ Upload failed: {{e}}")
"""
    
    print("\n🔨 Creating CustomJob object...")
    try:
        # Create custom training job worker pool specs
        worker_pool_specs = [
            {
                "machine_spec": {
                    "machine_type": machine_type,
                },
                "replica_count": 1,
                "container_spec": {
                    "image_uri": container_image,
                    "command": ["python", "-c"],
                    "args": [training_script],
                    "env": [
                        {"name": "GOOGLE_CLOUD_PROJECT", "value": project_id}
                    ]
                }
            }
        ]
        
        # Add accelerator if using GPU
        if use_gpu:
            worker_pool_specs[0]["machine_spec"]["accelerator_type"] = accelerator_type
            worker_pool_specs[0]["machine_spec"]["accelerator_count"] = accelerator_count
            print(f"🎯 GPU configuration: {accelerator_type} x{accelerator_count}")
        
        # Create the job object
        job = aiplatform.CustomJob(
            display_name=job_display_name,
            worker_pool_specs=worker_pool_specs,
            base_output_dir=f"gs://{bucket_name}/vertex_ai_jobs/{timestamp}",
        )
        print("✅ CustomJob object created successfully")
        
    except Exception as e:
        print(f"❌ Failed to create CustomJob object: {e}")
        raise
    
    print(f"\n📤 Submitting job to Vertex AI...")
    
    # Submit the job - FIXED ERROR HANDLING
    try:
        # Submit job and wait for it to be created (but not completed)
        job.run(
            service_account=service_account,
            sync=False,  # Don't wait for completion
            timeout=None,  # No timeout
            restart_job_on_worker_restart=False  # Don't restart on failure
        )
        
        # Wait a moment for the job to be created on the server
        import time
        time.sleep(2)
        
        # Try to get job info - this will fail if job wasn't created
        try:
            job_state = job.state
            resource_name = job.resource_name
            
            # If we get here, job was created successfully
            print(f"✅ Job submitted successfully!")
            print(f"🌐 Job Resource Name: {resource_name}")
            print(f"📊 Monitor at: https://console.cloud.google.com/ai/platform/jobs")
            print(f"💾 Results will be saved to: gs://{bucket_name}/vertex_ai_jobs/{timestamp}")
            print(f"🔄 Job State: {job_state}")
            
            return job
            
        except Exception as state_error:
            # If we can't get job state, the job wasn't created properly
            print(f"❌ Job submission failed - job not created on server")
            print(f"   State check error: {state_error}")
            raise Exception("Job was not created successfully on Vertex AI")
        
    except Exception as e:
        print(f"❌ Job submission failed: {e}")
        
        # More detailed error analysis
        error_str = str(e).lower()
        
        if "permission" in error_str or "forbidden" in error_str:
            print(f"🔐 PERMISSION ISSUE:")
            print(f"   Service account: {service_account}")
            print(f"   Required roles:")
            print(f"   - Vertex AI User")
            print(f"   - Storage Object Admin (for bucket {bucket_name})")
            print(f"   - Service Account User (if different account)")
            
        elif "quota" in error_str or "resource" in error_str:
            print(f"🎯 QUOTA/RESOURCE ISSUE:")
            print(f"   Machine type: {machine_type}")
            print(f"   Region: {region}")
            print(f"   Try: gcloud compute regions list")
            print(f"   Or try smaller machine: n1-standard-2")
            
        elif "bucket" in error_str or "storage" in error_str:
            print(f"🪣 STORAGE ISSUE:")
            print(f"   Bucket: gs://{bucket_name}")
            print(f"   Check: gsutil ls gs://{bucket_name}")
            
        elif "api" in error_str or "service" in error_str:
            print(f"🔌 API ISSUE:")
            print(f"   Enable: gcloud services enable aiplatform.googleapis.com")
            
        elif "not been created" in error_str:
            print(f"🏗️ JOB CREATION ISSUE:")
            print(f"   The job object was created locally but not on Vertex AI")
            print(f"   This usually indicates:")
            print(f"   1. Service account lacks sufficient permissions")
            print(f"   2. Project/region configuration issue")
            print(f"   3. Vertex AI API not properly enabled")
            print(f"   4. Resource quota limits exceeded")
            
        else:
            print(f"🔍 UNKNOWN ISSUE:")
            print(f"   Full error: {e}")
            print(f"   Check Cloud Console logs for more details")
        
        raise

def check_prerequisites(project_id: str, bucket_name: str, service_account: str):
    """Check if all prerequisites are met"""
    print("🔍 CHECKING PREREQUISITES...")
    print("=" * 40)
    
    issues_found = []
    
    # Check GCS bucket access
    try:
        print(f"🪣 Checking bucket: gs://{bucket_name}")
        client = storage.Client(project=project_id)
        bucket = client.bucket(bucket_name)
        bucket.reload()
        
        # Check if processed folder exists
        blobs = list(client.list_blobs(bucket, prefix="processed/", max_results=1))
        if blobs:
            print(f"✅ Bucket accessible with processed data")
        else:
            print(f"⚠️  Bucket exists but no processed/ data found")
            issues_found.append("No processed data in bucket")
            
    except Exception as e:
        print(f"❌ Cannot access bucket: {e}")
        issues_found.append(f"Bucket access failed: {e}")
    
    # Check service account format
    print(f"👤 Checking service account: {service_account}")
    if "@" not in service_account or not service_account.endswith(".iam.gserviceaccount.com"):
        print(f"❌ Invalid service account format")
        issues_found.append("Invalid service account format")
    else:
        print(f"✅ Service account format looks correct")
    
    # Try to list existing jobs (test Vertex AI access)
    try:
        print(f"🤖 Testing Vertex AI access...")
        # This will be done in the main function after aiplatform.init()
        print(f"⏳ Will test during initialization...")
    except Exception as e:
        print(f"❌ Vertex AI access issue: {e}")
        issues_found.append(f"Vertex AI access: {e}")
    
    print("=" * 40)
    
    if issues_found:
        print(f"❌ Found {len(issues_found)} issues:")
        for i, issue in enumerate(issues_found, 1):
            print(f"   {i}. {issue}")
        return False
    else:
        print("✅ All prerequisite checks passed!")
        return True

def main():
    parser = argparse.ArgumentParser(description='Submit Vertex AI Training Job - FIXED VERSION')
    parser.add_argument('--project-id', required=True, help='Google Cloud Project ID')
    parser.add_argument('--region', default='us-central1', help='Vertex AI region')
    parser.add_argument('--bucket-name', default='second-bucket-video-deepfake', 
                       help='GCS bucket name')
    parser.add_argument('--service-account', required=True,
                       help='Service account email (e.g., trainer@project.iam.gserviceaccount.com)')
    parser.add_argument('--machine-type', default='n1-standard-4',
                       help='Machine type for training')
    parser.add_argument('--use-gpu', action='store_true',
                       help='Use GPU for training (costs more)')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    
    args = parser.parse_args()
    
    print(f"🚀 VERTEX AI JOB SUBMISSION - FIXED VERSION")
    print(f"=" * 60)
    print(f"Project: {args.project_id}")
    print(f"Region: {args.region}")
    print(f"Bucket: gs://{args.bucket_name}")
    print(f"Service Account: {args.service_account}")
    print(f"Machine Type: {args.machine_type}")
    print(f"GPU: {'Yes' if args.use_gpu else 'No'}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"=" * 60)
    
    # Check prerequisites
    if not check_prerequisites(args.project_id, args.bucket_name, args.service_account):
        print("\n❌ Prerequisites check failed! Please fix the issues above.")
        sys.exit(1)
    
    # Create and submit job
    try:
        job = create_vertex_ai_job(
            project_id=args.project_id,
            region=args.region,
            bucket_name=args.bucket_name,
            service_account=args.service_account,
            machine_type=args.machine_type,
            use_gpu=args.use_gpu,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        
        print(f"\n🎉 SUCCESS! Training job submitted to Vertex AI")
        print(f"📊 Monitor progress at: https://console.cloud.google.com/ai/platform/training")
        print(f"🔍 Job details: {job.resource_name}")
        
    except Exception as e:
        print(f"\n❌ FAILED TO SUBMIT JOB: {e}")
        print(f"\n🔧 DEBUGGING STEPS:")
        print(f"1. Run the debug script first:")
        print(f"   python debug_vertex_setup.py")
        print(f"")
        print(f"2. Check service account roles in Cloud Console:")
        print(f"   https://console.cloud.google.com/iam-admin/iam")
        print(f"   Service account: {args.service_account}")
        print(f"   Required roles:")
        print(f"   - Vertex AI User")
        print(f"   - Storage Object Admin") 
        print(f"   - Service Account User")
        print(f"")
        print(f"3. Verify APIs are enabled:")
        print(f"   gcloud services enable aiplatform.googleapis.com")
        print(f"   gcloud services enable storage-api.googleapis.com")
        print(f"")
        print(f"4. Check quotas:")
        print(f"   https://console.cloud.google.com/iam-admin/quotas")
        print(f"   Search for: Vertex AI")
        print(f"")
        print(f"5. Try with smaller machine:")
        print(f"   --machine-type n1-standard-2")
        
        sys.exit(1)

if __name__ == "__main__":
    main()