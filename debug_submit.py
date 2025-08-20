#!/usr/bin/env python3
import sys
import argparse
from datetime import datetime
from google.cloud import aiplatform
from google.cloud import storage

def debug_submit_training(project_id: str):
    """Debug version with more error handling"""
    
    try:
        # Test storage access first
        print("🔍 Testing storage access...")
        storage_client = storage.Client(project=project_id)
        bucket = storage_client.bucket("second-bucket-video-deepfake")
        
        # Check if bucket exists
        if not bucket.exists():
            print("❌ Bucket 'second-bucket-video-deepfake' does not exist!")
            print("Creating bucket...")
            bucket = storage_client.create_bucket("second-bucket-video-deepfake", location="us-central1")
            print("✅ Bucket created")
        else:
            print("✅ Bucket exists")
        
        # Initialize Vertex AI with explicit project
        print("🔍 Initializing Vertex AI...")
        aiplatform.init(
            project=project_id,
            location="us-central1"
        )
        print("✅ Vertex AI initialized")
        
        # Simple job configuration
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        job_name = f"test-job-{timestamp}"
        
        print(f"📋 Creating job: {job_name}")
        
        # Minimal job spec
        job = aiplatform.CustomJob(
            display_name=job_name,
            worker_pool_specs=[
                {
                    "machine_spec": {"machine_type": "n1-standard-4"},
                    "replica_count": 1,
                    "container_spec": {
                        "image_uri": "gcr.io/cloud-aiplatform/training/pytorch-cpu.1-13:latest",
                        "command": ["echo"],
                        "args": ["Test job is running"]
                    }
                }
            ]
        )
        
        print("📤 Submitting job...")
        job.run(sync=False)
        print(f"✅ Job submitted: {job.resource_name}")
        
    except Exception as e:
        print(f"❌ Detailed error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-id', default='video-deep-fake-detector')
    args = parser.parse_args()
    
    debug_submit_training(args.project_id)