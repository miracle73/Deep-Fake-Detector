
"""
Submit Video Deepfake Training Job to Vertex AI
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
    
    # Initialize Vertex AI
    aiplatform.init(project=project_id, location=region)
    
    # Job display name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_display_name = f"deepfake-detector-{timestamp}"
    
    # Container image
    if use_gpu:
        container_image = "gcr.io/cloud-aiplatform/training/pytorch-gpu.1-13:latest"
        accelerator_type = "NVIDIA_TESLA_T4"
        accelerator_count = 1
    else:
        container_image = "gcr.io/cloud-aiplatform/training/pytorch-cpu.1-13:latest"
        accelerator_type = None
        accelerator_count = 0
    
    # Training arguments
    training_args = [
        f"--bucket-name={bucket_name}",
        f"--epochs={epochs}",
        f"--batch-size={batch_size}",
        "--max-frames=16",
        "--frame-encoder=mobilenet_v2",
        "--temporal-model=lstm",
        f"--gcs-prefix=models/experiments/{timestamp}"
    ]
    
    print("🚀 SUBMITTING VERTEX AI TRAINING JOB")
    print("=" * 50)
    print(f"📋 Job Name: {job_display_name}")
    print(f"🖥️  Machine Type: {machine_type}")
    print(f"🎯 GPU: {'Yes' if use_gpu else 'No'}")
    print(f"📊 Data: gs://{bucket_name}/processed/")
    print(f"⚙️  Args: {' '.join(training_args)}")
    print("=" * 50)
    
    # Create custom training job
    job = aiplatform.CustomJob(
        display_name=job_display_name,
        worker_pool_specs=[
            {
                "machine_spec": {
                    "machine_type": machine_type,
                    **({"accelerator_type": accelerator_type, 
                        "accelerator_count": accelerator_count} if use_gpu else {})
                },
                "replica_count": 1,
                "container_spec": {
                    "image_uri": container_image,
                    "command": ["python", "scripts/vertex_ai_training_cpu_optimized.py"],
                    "args": training_args,
                    "env": [
                        {"name": "GOOGLE_CLOUD_PROJECT", "value": project_id}
                    ]
                }
            }
        ],
        base_output_dir=f"gs://{bucket_name}/vertex_ai_jobs/{timestamp}",
        # staging_bucket=f"gs://{bucket_name}",  # Uncomment if needed
    )
    
    print(f"📤 Submitting job to Vertex AI...")
    
    # Submit the job
    job.run(
        service_account=service_account,
        sync=False  # Don't wait for completion
    )
    
    print(f"✅ Job submitted successfully!")
    print(f"🌐 Job ID: {job.resource_name}")
    print(f"📊 Monitor at: https://console.cloud.google.com/ai/platform/jobs")
    print(f"💾 Results will be saved to: gs://{bucket_name}/models/experiments/{timestamp}")
    
    return job

def check_prerequisites(project_id: str, bucket_name: str):
    """Check if all prerequisites are met"""
    print("🔍 Checking prerequisites...")
    
    # Check GCS bucket access
    try:
        client = storage.Client(project=project_id)
        bucket = client.bucket(bucket_name)
        
        # Check if processed data exists
        blobs = list(client.list_blobs(bucket, prefix="processed/", max_results=10))
        if not blobs:
            print("❌ No processed data found in bucket!")
            return False
            
        print(f"✅ Found processed data in gs://{bucket_name}/processed/")
        
    except Exception as e:
        print(f"❌ Cannot access bucket {bucket_name}: {e}")
        return False
    
    # Check Vertex AI API
    try:
        aiplatform.init(project=project_id, location="us-central1")
        print("✅ Vertex AI API accessible")
    except Exception as e:
        print(f"❌ Cannot access Vertex AI: {e}")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Submit Vertex AI Training Job')
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
    
    # Check prerequisites
    if not check_prerequisites(args.project_id, args.bucket_name):
        print("❌ Prerequisites check failed!")
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
        print(f"Monitor progress at: https://console.cloud.google.com/ai/platform/training")
        
    except Exception as e:
        print(f"❌ Failed to submit job: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()