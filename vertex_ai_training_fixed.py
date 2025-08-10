#!/usr/bin/env python3
"""
vertex_ai_training_fixed.py - Train deepfake detector on Vertex AI with 297K images
FIXED: Properly handle job submission and resource creation + CPU-only support
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
import time

class VertexAIDeepfakeTrainer:
    def __init__(self, project_id: str, bucket_name: str, region: str, credentials_path: str = None):
        """Initialize Vertex AI trainer with proper authentication"""
        self.project_id = project_id
        self.bucket_name = bucket_name
        self.region = region
        
        # Handle authentication
        self.credentials = None
        credentials_file = None
        
        # Priority order for credentials
        if credentials_path and os.path.exists(credentials_path):
            credentials_file = credentials_path
            print(f"🔐 Using explicit credentials: {credentials_path}")
        elif os.getenv('GOOGLE_VERTEX_AI_APPLICATION_CREDENTIALS'):
            credentials_file = os.getenv('GOOGLE_VERTEX_AI_APPLICATION_CREDENTIALS')
            if os.path.exists(credentials_file):
                print(f"🔐 Using GOOGLE_VERTEX_AI_APPLICATION_CREDENTIALS: {credentials_file}")
            else:
                print(f"❌ GOOGLE_VERTEX_AI_APPLICATION_CREDENTIALS points to non-existent file: {credentials_file}")
                credentials_file = None
        elif os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
            credentials_file = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
            if os.path.exists(credentials_file):
                print(f"🔐 Using GOOGLE_APPLICATION_CREDENTIALS: {credentials_file}")
            else:
                print(f"❌ GOOGLE_APPLICATION_CREDENTIALS points to non-existent file: {credentials_file}")
                credentials_file = None
        else:
            print("🔐 Attempting to use default credentials")
        
        # Load credentials if we found a file
        if credentials_file:
            try:
                self.credentials = service_account.Credentials.from_service_account_file(
                    credentials_file,
                    scopes=["https://www.googleapis.com/auth/cloud-platform"]
                )
                print(f"✅ Successfully loaded service account credentials")
            except Exception as e:
                print(f"❌ Failed to load credentials from {credentials_file}: {e}")
                self.credentials = None
        
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
            client = storage.Client(credentials=self.credentials, project=self.project_id)
            bucket = client.bucket(self.bucket_name)
            
            if bucket.exists():
                print(f"✅ Authentication test passed - can access bucket: {self.bucket_name}")
                return True
            else:
                print(f"❌ Bucket {self.bucket_name} not found or not accessible")
                return False
                
        except Exception as e:
            print(f"❌ Authentication test failed: {e}")
            print("💡 Try one of these solutions:")
            print("   1. Set GOOGLE_APPLICATION_CREDENTIALS environment variable")
            print("   2. Use --credentials-path parameter")
            print("   3. Run 'gcloud auth application-default login'")
            return False

    def create_training_config(self, use_gpu: bool = True):
        """Create optimized training configuration for large dataset"""
        config = {
            'model': {
                'name': 'efficientnet_b4',
                'num_classes': 2,
                'pretrained': True,
                'dropout': 0.3
            },
            'training': {
                'batch_size': 16 if not use_gpu else 32,  # Smaller batch for CPU
                'epochs': 50,
                'learning_rate': 0.0001,
                'weight_decay': 0.0001,
                'patience': 10,
                'gradient_accumulation_steps': 4 if not use_gpu else 2,  # More accumulation for CPU
                'mixed_precision': use_gpu,  # Only use mixed precision with GPU
                'num_workers': 2 if not use_gpu else 4,
                'device': 'cuda' if use_gpu else 'cpu'
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
    
    def submit_training_job(self, machine_type: str = "n1-highmem-8", 
                           accelerator_type: str = "NVIDIA_TESLA_T4",
                           accelerator_count: int = 1):
        """Submit training job to Vertex AI - FIXED VERSION with CPU support"""
        
        print(f"\n🚀 Submitting training job: {self.job_name}")
        
        # Test authentication first
        if not self.test_authentication():
            print("❌ Authentication test failed. Please check your credentials.")
            return None
        
        # Determine if using GPU
        use_gpu = accelerator_count > 0
        
        # Create training config
        config = self.create_training_config(use_gpu=use_gpu)
        
        # Use existing Docker image
        image_uri = f"gcr.io/{self.project_id}/deepfake-detector:20250809_175159"
        print(f"♻️  Using existing Docker image: {image_uri}")
        
        # Create machine spec - conditionally include GPU settings
        machine_spec = {
            "machine_type": machine_type,
        }
        
        # Only add GPU settings if we're using GPU
        if use_gpu:
            machine_spec["accelerator_type"] = accelerator_type
            machine_spec["accelerator_count"] = accelerator_count
            print(f"🔥 Using GPU: {accelerator_count}x {accelerator_type}")
        else:
            print(f"💻 Using CPU-only training")
        
        # Define worker pool specs
        worker_pool_specs = [
            {
                "machine_spec": machine_spec,
                "replica_count": 1,
                "container_spec": {
                    "image_uri": image_uri,
                    "command": ["python", "scripts/train_model.py"],
                    "args": [
                        "--config", "/app/config/config.yaml",
                        "--data-dir", "/gcs/deepfake_dataset"
                    ],
                    "env": [
                        {"name": "TRAINING_CONFIG", "value": json.dumps(config)},
                        {"name": "BUCKET_NAME", "value": self.bucket_name},
                        {"name": "JOB_NAME", "value": self.job_name},
                        {"name": "PYTHONUNBUFFERED", "value": "1"},
                        {"name": "USE_GPU", "value": str(use_gpu).lower()},
                    ],
                },
            }
        ]
        
        try:
            print(f"📊 Creating custom training job...")
            print(f"   Machine: {machine_type}")
            if use_gpu:
                print(f"   GPU: {accelerator_count}x {accelerator_type}")
            else:
                print(f"   GPU: None (CPU-only)")
            print(f"   Dataset: ~297K images from GCS bucket")
            
            # Create the custom job
            job = aiplatform.CustomJob(
                display_name=self.job_name,
                worker_pool_specs=worker_pool_specs,
                staging_bucket=f"gs://{self.bucket_name}/vertex_ai_staging",
            )
            
            print(f"✅ Job created, now submitting...")
            
            # Submit the job - using the simple submit() without extra parameters
            job.submit()
            
            # Wait a moment for the job to be fully created
            print("⏳ Waiting for job to initialize...")
            time.sleep(5)
            
            # Now the job should be created and we can access its properties
            print(f"\n✅ Job submitted successfully!")
            print(f"   Job Name: {job.display_name}")
            print(f"   Job ID: {job.name}")
            print(f"   State: {job.state}")
            print(f"\n📊 Monitor your job at:")
            print(f"   https://console.cloud.google.com/vertex-ai/training/custom-jobs?project={self.project_id}")
            print(f"\n💡 To check job status from command line:")
            print(f"   gcloud ai custom-jobs describe {job.name.split('/')[-1]} --region={self.region}")
            print(f"\n📁 Output will be saved to:")
            print(f"   gs://{self.bucket_name}/models/{self.job_name}/")
            
            if not use_gpu:
                print(f"\n⚠️  CPU-only training will be MUCH slower than GPU training.")
                print(f"   Consider requesting T4 GPU quota increase for faster training.")
            
            return job
            
        except Exception as e:
            print(f"❌ Failed to submit job: {e}")
            print(f"\n🔍 Debugging info:")
            print(f"   Error type: {type(e).__name__}")
            print(f"   Error details: {str(e)}")
            
            # Try to provide more specific guidance based on the error
            if "not enabled" in str(e).lower():
                print(f"\n💡 Make sure Vertex AI API is enabled:")
                print(f"   gcloud services enable aiplatform.googleapis.com --project={self.project_id}")
            elif "permission" in str(e).lower() or "forbidden" in str(e).lower():
                print(f"\n💡 Check that your service account has the required permissions:")
                print(f"   - Vertex AI User")
                print(f"   - Storage Object Admin")
                print(f"   - Service Account User")
            elif "quota" in str(e).lower():
                print(f"\n💡 You may have exceeded your quota. Check:")
                print(f"   https://console.cloud.google.com/iam-admin/quotas?project={self.project_id}")
            
            return None

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Train Deepfake Detector on Vertex AI')
    parser.add_argument('--machine-type', default='n1-highmem-8', help='Machine type')
    parser.add_argument('--gpu-type', default='NVIDIA_TESLA_T4', help='GPU type')
    parser.add_argument('--gpu-count', type=int, default=1, help='Number of GPUs (set to 0 for CPU-only)')
    parser.add_argument('--project-id', required=True, help='GCP Project ID')
    parser.add_argument('--bucket-name', required=True, help='GCS Bucket name')
    parser.add_argument('--region', default='us-central1', help='GCP Region')
    parser.add_argument('--credentials-path', help='Path to service account JSON key file')
    
    args = parser.parse_args()
    
    print(f"🚀 VERTEX AI DEEPFAKE DETECTOR TRAINING")
    print(f"=" * 50)
    print(f"   Project: {args.project_id}")
    print(f"   Bucket: {args.bucket_name}")
    print(f"   Region: {args.region}")
    
    # Show current environment variables for debugging
    print(f"\n🔍 Environment Check:")
    if os.getenv('GOOGLE_VERTEX_AI_APPLICATION_CREDENTIALS'):
        cred_file = os.getenv('GOOGLE_VERTEX_AI_APPLICATION_CREDENTIALS')
        exists = os.path.exists(cred_file)
        print(f"   GOOGLE_VERTEX_AI_APPLICATION_CREDENTIALS: {cred_file} ({'✅' if exists else '❌'})")
    
    if os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
        cred_file = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        exists = os.path.exists(cred_file)
        print(f"   GOOGLE_APPLICATION_CREDENTIALS: {cred_file} ({'✅' if exists else '❌'})")
    
    if not os.getenv('GOOGLE_VERTEX_AI_APPLICATION_CREDENTIALS') and not os.getenv('GOOGLE_APPLICATION_CREDENTIALS') and not args.credentials_path:
        print(f"   ⚠️  No credentials found. Will try default credentials.")
    
    # Initialize trainer
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
        print("\n✅ SUCCESS! Your training job is running.")
        print("\n🎯 Next Steps:")
        print("1. Monitor job progress in Cloud Console (link above)")
        print("2. Check logs for training metrics")
        print("3. Wait for training to complete (may take several hours)")
        print("4. Download trained model from GCS when complete")
        
        if args.gpu_count == 0:
            print("\n⏰ CPU Training Time Estimates:")
            print("   - With 297K images, CPU training may take 10-20+ hours")
            print("   - Consider requesting GPU quota for much faster training")
    else:
        print("\n❌ Job submission failed!")
        print("Please check the error messages above and try again.")

if __name__ == "__main__":
    main()