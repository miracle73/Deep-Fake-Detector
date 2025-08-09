#!/usr/bin/env python3
"""
upload_data_to_gcs.py - Efficient parallel upload of deepfake dataset to GCS
"""

import os
import sys
from pathlib import Path
from google.cloud import storage
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
import json
from typing import List, Tuple
import hashlib

class DataUploader:
    def __init__(self, bucket_name: str, project_id: str, credentials_path: str = None):
        """Initialize GCS uploader"""
        self.bucket_name = bucket_name
        self.project_id = project_id
        
        # Initialize GCS client with explicit credentials if provided
        if credentials_path and os.path.exists(credentials_path):
            self.storage_client = storage.Client.from_service_account_json(
                credentials_path, project=project_id
            )
            print(f"✅ Using service account credentials: {credentials_path}")
        else:
            # Try to use default credentials
            try:
                self.storage_client = storage.Client(project=project_id)
                print("✅ Using default credentials")
            except Exception as e:
                print(f"❌ Failed to initialize GCS client: {e}")
                print("💡 Please set up authentication:")
                print("   1. Set GOOGLE_APPLICATION_CREDENTIALS environment variable")
                print("   2. Run 'gcloud auth application-default login'")
                print("   3. Or provide credentials_path parameter")
                sys.exit(1)
        
        self.bucket = self.storage_client.bucket(bucket_name)
        
        # Upload statistics
        self.upload_stats = {
            'total_files': 0,
            'uploaded_files': 0,
            'skipped_files': 0,
            'failed_files': 0,
            'total_size_mb': 0
        }
    
    def check_file_exists(self, gcs_path: str) -> bool:
        """Check if file already exists in GCS"""
        blob = self.bucket.blob(gcs_path)
        return blob.exists()
    
    def upload_file(self, local_path: Path, gcs_path: str) -> Tuple[bool, str]:
        """Upload single file to GCS"""
        try:
            # Check if already exists (skip if identical)
            if self.check_file_exists(gcs_path):
                return True, "skipped"
            
            # Upload file
            blob = self.bucket.blob(gcs_path)
            blob.upload_from_filename(str(local_path))
            
            return True, "uploaded"
        
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    def upload_dataset_parallel(self, data_dir: str = "data", max_workers: int = 10):
        """Upload entire dataset to GCS with parallel processing"""
        data_path = Path(data_dir)
        
        # Collect all image files
        print("📊 Scanning dataset...")
        all_files = []
        
        # Your structure: data/raw/real/ and data/raw/fake/
        for category in ['real', 'fake']:
            category_dir = data_path / 'raw' / category
            if category_dir.exists():
                print(f"Found {category} directory")
                # Check for different image formats
                for pattern in ['*.jpg', '*.jpeg', '*.png']:
                    for img_file in category_dir.glob(pattern):
                        # Upload to GCS maintaining the category structure
                        gcs_path = f"deepfake_dataset/{category}/{img_file.name}"
                        all_files.append((img_file, gcs_path))
                        
                # Count files in this directory
                file_count = len(list(category_dir.glob('*.*')))
                print(f"  → Found {file_count} files in {category}/")
            else:
                print(f"❌ {category} directory not found at {category_dir}")
        
        self.upload_stats['total_files'] = len(all_files)
        print(f"Found {len(all_files):,} files to upload")
        
        # Upload files in parallel
        print(f"🚀 Starting parallel upload with {max_workers} workers...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all upload tasks
            future_to_file = {
                executor.submit(self.upload_file, local_path, gcs_path): (local_path, gcs_path)
                for local_path, gcs_path in all_files
            }
            
            # Process completed uploads
            with tqdm(total=len(all_files), desc="Uploading") as pbar:
                for future in as_completed(future_to_file):
                    local_path, gcs_path = future_to_file[future]
                    
                    try:
                        success, status = future.result()
                        
                        if success:
                            if status == "uploaded":
                                self.upload_stats['uploaded_files'] += 1
                                file_size_mb = local_path.stat().st_size / (1024 * 1024)
                                self.upload_stats['total_size_mb'] += file_size_mb
                            elif status == "skipped":
                                self.upload_stats['skipped_files'] += 1
                        else:
                            self.upload_stats['failed_files'] += 1
                            print(f"❌ Failed: {gcs_path} - {status}")
                    
                    except Exception as e:
                        self.upload_stats['failed_files'] += 1
                        print(f"❌ Exception: {gcs_path} - {e}")
                    
                    pbar.update(1)
                    pbar.set_postfix({
                        'Uploaded': self.upload_stats['uploaded_files'],
                        'Skipped': self.upload_stats['skipped_files'],
                        'Failed': self.upload_stats['failed_files']
                    })
        
        # Print summary
        print("\n📈 Upload Summary:")
        print(f"   Total Files: {self.upload_stats['total_files']:,}")
        print(f"   Uploaded: {self.upload_stats['uploaded_files']:,}")
        print(f"   Skipped: {self.upload_stats['skipped_files']:,}")
        print(f"   Failed: {self.upload_stats['failed_files']:,}")
        print(f"   Total Size: {self.upload_stats['total_size_mb']:.2f} MB")
        
        # Save manifest
        self.save_manifest()
        
        return self.upload_stats
    
    def save_manifest(self):
        """Save dataset manifest to GCS"""
        manifest = {
            'dataset_info': {
                'total_images': self.upload_stats['total_files'],
                'uploaded': self.upload_stats['uploaded_files'],
                'size_mb': self.upload_stats['total_size_mb'],
                'timestamp': time.time()
            },
            'categories': {
                'real': 0,
                'fake': 0
            }
        }
        
        # Count images per category
        for category in ['real', 'fake']:
            prefix = f"deepfake_dataset/{category}/"
            blobs = self.bucket.list_blobs(prefix=prefix)
            count = sum(1 for _ in blobs)
            manifest['categories'][category] = count
        
        # Upload manifest
        manifest_blob = self.bucket.blob('deepfake_dataset/manifest.json')
        manifest_blob.upload_from_string(json.dumps(manifest, indent=2))
        
        print(f"✅ Manifest saved to gs://{self.bucket_name}/deepfake_dataset/manifest.json")

def main():
    """Main upload function"""
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    project_id = os.getenv('PROJECT_ID')
    bucket_name = os.getenv('BUCKET_NAME')
    credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    
    if not project_id or not bucket_name:
        print("❌ Please set PROJECT_ID and BUCKET_NAME in .env file")
        sys.exit(1)
    
    print(f"🚀 Uploading dataset to gs://{bucket_name}")
    print(f"   Project: {project_id}")
    
    # Initialize uploader
    uploader = DataUploader(bucket_name, project_id, credentials_path)
    
    # Upload dataset
    stats = uploader.upload_dataset_parallel(
        data_dir="data",
        max_workers=20  # Adjust based on network speed
    )
    
    print("\n✅ Upload complete!")
    print(f"   Dataset available at: gs://{bucket_name}/deepfake_dataset/")

if __name__ == "__main__":
    main()