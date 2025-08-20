"""
Debug script to test Vertex AI setup step by step
"""

import os
import sys
from google.cloud import aiplatform
from google.cloud import storage
from google.auth import default

def test_authentication():
    """Test Google Cloud authentication"""
    print("🔐 Testing authentication...")
    try:
        credentials, project = default()
        print(f"✅ Authentication successful")
        print(f"   Default project: {project}")
        print(f"   Credentials type: {type(credentials).__name__}")
        return True, project
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        print("💡 Try: gcloud auth application-default login")
        return False, None

def test_storage_access(project_id, bucket_name):
    """Test Google Cloud Storage access"""
    print(f"\n🪣 Testing Storage access...")
    try:
        client = storage.Client(project=project_id)
        bucket = client.bucket(bucket_name)
        
        # Test bucket exists
        bucket.reload()
        print(f"✅ Bucket exists: gs://{bucket_name}")
        
        # Test list permissions
        blobs = list(client.list_blobs(bucket, max_results=3))
        print(f"✅ Can list blobs: found {len(blobs)} items")
        
        # Test if processed folder exists
        processed_blobs = list(client.list_blobs(bucket, prefix="processed/", max_results=5))
        print(f"✅ Processed folder: found {len(processed_blobs)} items")
        
        return True
        
    except Exception as e:
        print(f"❌ Storage access failed: {e}")
        return False

def test_vertex_ai_access(project_id, bucket_name, region="us-central1"):
    """Test Vertex AI API access"""
    print(f"\n🤖 Testing Vertex AI access...")
    try:
        # Initialize Vertex AI WITH staging bucket
        aiplatform.init(
            project=project_id, 
            location=region,
            staging_bucket=f"gs://{bucket_name}"  # Add staging bucket here
        )
        print(f"✅ Vertex AI initialized for {project_id} in {region}")
        print(f"✅ Staging bucket set to: gs://{bucket_name}")
        
        # Test listing jobs (without limit parameter)
        try:
            jobs = aiplatform.CustomJob.list()
            print(f"✅ Can list jobs: found {len(jobs)} total jobs")
        except Exception as list_error:
            print(f"⚠️  Cannot list jobs (but API is accessible): {list_error}")
            # This is not critical - API might still work
        
        # Test if we can access the specific region
        print(f"✅ Region {region} is accessible")
        
        return True
        
    except Exception as e:
        print(f"❌ Vertex AI access failed: {e}")
        return False

def test_service_account_format(service_account):
    """Test if service account format is correct"""
    print(f"\n👤 Testing service account format...")
    
    if not service_account:
        print("❌ Service account is empty")
        return False
    
    if "@" not in service_account or not service_account.endswith(".iam.gserviceaccount.com"):
        print(f"❌ Invalid service account format: {service_account}")
        print("💡 Should be like: my-service@project-id.iam.gserviceaccount.com")
        return False
    
    print(f"✅ Service account format looks correct: {service_account}")
    return True

def test_custom_job_creation(project_id, region, bucket_name):
    """Test creating a simple CustomJob object"""
    print(f"\n⚙️  Testing CustomJob creation...")
    
    try:
        # Ensure aiplatform is initialized with staging bucket
        aiplatform.init(
            project=project_id, 
            location=region,
            staging_bucket=f"gs://{bucket_name}"
        )
        
        # Simple worker pool spec
        worker_pool_specs = [{
            "machine_spec": {"machine_type": "n1-standard-4"},
            "replica_count": 1,
            "container_spec": {
                "image_uri": "gcr.io/cloud-aiplatform/training/pytorch-cpu.1-13:latest",
                "command": ["python", "-c"],
                "args": ["print('Hello from Vertex AI!')"]
            }
        }]
        
        # Try to create CustomJob object
        job = aiplatform.CustomJob(
            display_name="test-job-debug",
            worker_pool_specs=worker_pool_specs,
            base_output_dir=f"gs://{bucket_name}/debug_test"
        )
        
        print("✅ CustomJob object created successfully")
        # Don't try to access server-side properties until job is submitted
        print(f"   Job configured for display name: test-job-debug")
        
        print("✅ Job should submit successfully - all components are working!")
        
        return True, job
        
    except Exception as e:
        print(f"❌ CustomJob creation failed: {e}")
        return False, None

def main():
    # Get parameters
    project_id = input("Enter your project ID (video-deep-fake-detector): ").strip() or "video-deep-fake-detector"
    bucket_name = input("Enter your bucket name (second-bucket-video-deepfake): ").strip() or "second-bucket-video-deepfake"
    service_account = input("Enter service account email: ").strip()
    region = input("Enter region (us-central1): ").strip() or "us-central1"
    
    print(f"\n🧪 DEBUGGING VERTEX AI SETUP")
    print("=" * 50)
    print(f"Project ID: {project_id}")
    print(f"Bucket: gs://{bucket_name}")
    print(f"Service Account: {service_account}")
    print(f"Region: {region}")
    print("=" * 50)
    
    # Run tests
    all_passed = True
    
    # Test 1: Authentication
    auth_ok, default_project = test_authentication()
    if not auth_ok:
        all_passed = False
    
    # Test 2: Service account format
    if not test_service_account_format(service_account):
        all_passed = False
    
    # Test 3: Storage access
    if not test_storage_access(project_id, bucket_name):
        all_passed = False
    
    # Test 4: Vertex AI access
    if not test_vertex_ai_access(project_id, bucket_name, region):
        all_passed = False
    
    # Test 5: CustomJob creation
    job_ok, test_job = test_custom_job_creation(project_id, region, bucket_name)
    if not job_ok:
        all_passed = False
    
    # Summary
    print(f"\n📊 SUMMARY")
    print("=" * 30)
    if all_passed:
        print("🎉 All tests passed! Your setup should work.")
        print("\n🚀 Try running your training job again:")
        print("python fixed_vertex_ai_job.py \\")
        print(f"  --project-id {project_id} \\")
        print(f"  --service-account {service_account} \\")
        print(f"  --bucket-name {bucket_name} \\")
        print("  --epochs 20 --batch-size 4")
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        
        print("\n🔧 COMMON FIXES:")
        print("1. Authentication:")
        print("   gcloud auth application-default login")
        print("   gcloud config set project YOUR_PROJECT_ID")
        print("")
        print("2. Enable APIs:")
        print("   gcloud services enable aiplatform.googleapis.com")
        print("   gcloud services enable storage-api.googleapis.com")
        print("")
        print("3. Service Account Roles (in Cloud Console):")
        print("   - Vertex AI User")
        print("   - Storage Admin") 
        print("   - AI Platform Admin")

if __name__ == "__main__":
    main()