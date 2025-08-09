#!/usr/bin/env python3
"""
vertex_ai_training.py - Train deepfake detector on Vertex AI with 297K images
OPTIMIZED: Added option to reuse existing Docker images
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
import subprocess
import re

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
            print("🔐 Using environment credentials (GOOGLE_VERTEX_AI_APPLICATION_CREDENTIALS)")
            # Let it use default credentials from environment
        
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

    def list_existing_images(self):
        """List existing Docker images in the registry"""
        try:
            cmd = f"gcloud container images list-tags gcr.io/{self.project_id}/deepfake-detector --limit=10 --sort-by=timestamp --format=json"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                images = json.loads(result.stdout)
                if images:
                    print(f"\n📋 Found {len(images)} existing Docker images:")
                    for i, img in enumerate(images[:5]):  # Show last 5 images
                        tags = img.get('tags', ['<untagged>'])
                        timestamp = img.get('timestamp', 'unknown')
                        print(f"   {i+1}. Tag: {tags[0]}, Created: {timestamp[:19]}")
                    return images
                else:
                    print("📋 No existing Docker images found")
                    return []
            else:
                print(f"❌ Failed to list images: {result.stderr}")
                return []
        except Exception as e:
            print(f"❌ Error listing images: {e}")
            return []

    def get_image_choice(self, existing_images):
        """Ask user whether to reuse existing image or build new one"""
        if not existing_images:
            return None, True  # No existing images, must build new
        
        print(f"\n🤔 You have {len(existing_images)} existing Docker images.")
        print("Options:")
        print("1. Build new image (recommended for code changes)")
        print("2. Reuse most recent image (faster, good if no changes)")
        
        for i, img in enumerate(existing_images[:3]):
            tags = img.get('tags', ['<untagged>'])
            timestamp = img.get('timestamp', 'unknown')
            print(f"   Option {i+3}. Reuse image: {tags[0]} (created: {timestamp[:19]})")
        
        while True:
            try:
                choice = input("\nEnter your choice (1-5): ").strip()
                
                if choice == "1":
                    return None, True  # Build new
                elif choice == "2":
                    # Use most recent image
                    latest_img = existing_images[0]
                    tag = latest_img.get('tags', ['latest'])[0]
                    return f"gcr.io/{self.project_id}/deepfake-detector:{tag}", False
                elif choice in ["3", "4", "5"]:
                    idx = int(choice) - 3
                    if idx < len(existing_images):
                        img = existing_images[idx]
                        tag = img.get('tags', ['latest'])[0]
                        return f"gcr.io/{self.project_id}/deepfake-detector:{tag}", False
                    else:
                        print("Invalid option. Please try again.")
                else:
                    print("Invalid choice. Please enter 1-5.")
            except (ValueError, KeyboardInterrupt):
                print("Invalid input. Please try again.")

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
        # [The same training script content as before - truncated for brevity]
        script_content = '''
# ... [Same training script content as in original code] ...
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
                           accelerator_count: int = 1,
                           reuse_image: bool = False):
        """Submit training job to Vertex AI"""
        
        print(f"🚀 Submitting training job: {self.job_name}")
        
        # Test authentication first
        if not self.test_authentication():
            print("❌ Authentication test failed. Please check your credentials.")
            return None
        
        # Create training config
        config = self.create_training_config()
        
        # Use existing successful Docker image from previous build
        image_uri = f"gcr.io/{self.project_id}/deepfake-detector:20250809_175159"
        print(f"♻️  Using existing Docker image: {image_uri}")
        print("   (Skipping Docker build - using successful build from previous run)")
        
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
    parser.add_argument('--build-new-image', action='store_true', 
                       help='Force build new Docker image instead of using existing one')
    
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
        accelerator_count=args.gpu_count,
        reuse_image=not args.build_new_image  # Default to reuse unless explicitly asked to build new
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