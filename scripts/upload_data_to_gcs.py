#!/usr/bin/env python3
"""
Upload video datasets to Google Cloud Storage
Optimized for large video files with progress tracking
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Optional
import json
from datetime import datetime

from google.cloud import storage
from google.auth import default
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoDataUploader:
    """Upload video datasets to Google Cloud Storage"""
    
    def __init__(
        self,
        project_id: str,
        bucket_name: str,
        credentials_path: Optional[str] = None
    ):
        self.project_id = project_id
        self.bucket_name = bucket_name
        
        # Initialize GCS client
        if credentials_path and Path(credentials_path).exists():
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
        
        try:
            self.client = storage.Client(project=project_id)
            self.bucket = self.client.bucket(bucket_name)
            logger.info(f"✅ Connected to GCS bucket: {bucket_name}")
        except Exception as e:
            logger.error(f"❌ Failed to connect to GCS: {e}")
            raise
    
    def create_bucket_if_not_exists(self, location: str = "US-CENTRAL1"):
        """Create bucket if it doesn't exist"""
        try:
            self.bucket = self.client.get_bucket(self.bucket_name)
            logger.info(f"✅ Bucket {self.bucket_name} already exists")
        except Exception:
            logger.info(f"Creating bucket {self.bucket_name}...")
            self.bucket = self.client.create_bucket(
                self.bucket_name,
                location=location
            )
            logger.info(f"✅ Created bucket {self.bucket_name}")
    
    def get_video_files(self, data_dir: str) -> List[tuple]:
        """Get all video files with their relative paths"""
        data_path = Path(data_dir)
        video_files = []
        
        # Video file extensions
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
        
        for file_path in data_path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in video_extensions:
                # Get relative path from data_dir
                relative_path = file_path.relative_to(data_path)
                video_files.append((str(file_path), str(relative_path)))
        
        return video_files
    
    def upload_file(
        self,
        local_path: str,
        gcs_path: str,
        chunk_size: int = 8 * 1024 * 1024  # 8MB chunks
    ) -> bool:
        """Upload a single file to GCS with progress tracking"""
        try:
            blob = self.bucket.blob(gcs_path)
            
            # Check if file already exists
            if blob.exists():
                logger.info(f"⏭️ Skipping existing file: {gcs_path}")
                return True
            
            # Get file size for progress tracking
            file_size = Path(local_path).stat().st_size
            file_size_mb = file_size / (1024 * 1024)
            
            logger.info(f"📤 Uploading {gcs_path} ({file_size_mb:.1f} MB)...")
            
            # Upload with progress tracking for large files
            if file_size > chunk_size:
                with open(local_path, 'rb') as file_obj:
                    with tqdm(total=file_size, unit='B', unit_scale=True, desc="Uploading") as pbar:
                        def progress_callback(bytes_transferred):
                            pbar.update(bytes_transferred - pbar.n)
                        
                        blob.upload_from_file(
                            file_obj,
                            size=file_size,
                            chunk_size=chunk_size,
                            progress_callback=progress_callback
                        )
            else:
                blob.upload_from_filename(local_path)
            
            # Set metadata
            blob.metadata = {
                'uploaded_at': datetime.now().isoformat(),
                'original_path': local_path,
                'file_size_bytes': str(file_size),
                'content_type': 'video/mp4'  # Default to mp4
            }
            blob.patch()
            
            logger.info(f"✅ Uploaded: {gcs_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to upload {local_path}: {e}")
            return False
    
    def upload_dataset(
        self,
        data_dir: str,
        gcs_prefix: str = "datasets/videos",
        max_workers: int = 4
    ) -> dict:
        """Upload entire video dataset to GCS"""
        logger.info(f"🚀 Starting upload from {data_dir} to gs://{self.bucket_name}/{gcs_prefix}")
        
        # Get all video files
        video_files = self.get_video_files(data_dir)
        
        if not video_files:
            logger.warning(f"⚠️ No video files found in {data_dir}")
            return {'success': 0, 'failed': 0, 'skipped': 0}
        
        logger.info(f"📹 Found {len(video_files)} video files")
        
        # Upload statistics
        stats = {'success': 0, 'failed': 0, 'skipped': 0}
        
        # Upload files
        for local_path, relative_path in tqdm(video_files, desc="Uploading videos"):
            gcs_path = f"{gcs_prefix}/{relative_path}"
            
            # Check if already exists
            blob = self.bucket.blob(gcs_path)
            if blob.exists():
                stats['skipped'] += 1
                continue
            
            success = self.upload_file(local_path, gcs_path)
            if success:
                stats['success'] += 1
            else:
                stats['failed'] += 1
        
        return stats
    
    def create_dataset_manifest(
        self,
        gcs_prefix: str = "datasets/videos",
        manifest_name: str = "dataset_manifest.json"
    ):
        """Create a manifest file listing all uploaded videos"""
        logger.info("📋 Creating dataset manifest...")
        
        # List all blobs with the prefix
        blobs = self.client.list_blobs(self.bucket, prefix=gcs_prefix)
        
        manifest = {
            'created_at': datetime.now().isoformat(),
            'bucket': self.bucket_name,
            'prefix': gcs_prefix,
            'videos': []
        }
        
        for blob in blobs:
            if not blob.name.endswith('/'):  # Skip directories
                video_info = {
                    'gcs_path': f"gs://{self.bucket_name}/{blob.name}",
                    'name': blob.name,
                    'size_bytes': blob.size,
                    'updated': blob.updated.isoformat() if blob.updated else None,
                    'md5_hash': blob.md5_hash
                }
                
                # Add metadata if available
                if blob.metadata:
                    video_info['metadata'] = blob.metadata
                
                manifest['videos'].append(video_info)
        
        # Upload manifest
        manifest_blob = self.bucket.blob(f"{gcs_prefix}/{manifest_name}")
        manifest_blob.upload_from_string(
            json.dumps(manifest, indent=2),
            content_type='application/json'
        )
        
        logger.info(f"✅ Created manifest with {len(manifest['videos'])} videos")
        logger.info(f"📄 Manifest uploaded to: gs://{self.bucket_name}/{gcs_prefix}/{manifest_name}")
    
    def validate_uploads(self, data_dir: str, gcs_prefix: str = "datasets/videos") -> dict:
        """Validate that all local files were uploaded correctly"""
        logger.info("🔍 Validating uploads...")
        
        local_files = self.get_video_files(data_dir)
        validation_results = {'missing': [], 'size_mismatch': [], 'valid': 0}
        
        for local_path, relative_path in local_files:
            gcs_path = f"{gcs_prefix}/{relative_path}"
            blob = self.bucket.blob(gcs_path)
            
            if not blob.exists():
                validation_results['missing'].append(relative_path)
                continue
            
            # Check file size
            local_size = Path(local_path).stat().st_size
            blob.reload()
            
            if blob.size != local_size:
                validation_results['size_mismatch'].append({
                    'file': relative_path,
                    'local_size': local_size,
                    'gcs_size': blob.size
                })
                continue
            
            validation_results['valid'] += 1
        
        return validation_results

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Upload video dataset to Google Cloud Storage')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Local directory containing video data')
    parser.add_argument('--project-id', type=str, required=True,
                       help='Google Cloud project ID')
    parser.add_argument('--bucket-name', type=str, required=True,
                       help='GCS bucket name')
    parser.add_argument('--gcs-prefix', type=str, default='datasets/videos',
                       help='GCS prefix for uploaded files')
    parser.add_argument('--credentials', type=str,
                       help='Path to service account credentials JSON')
    parser.add_argument('--create-bucket', action='store_true',
                       help='Create bucket if it does not exist')
    parser.add_argument('--validate', action='store_true',
                       help='Validate uploads after completion')
    parser.add_argument('--create-manifest', action='store_true',
                       help='Create dataset manifest file')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.data_dir).exists():
        logger.error(f"❌ Data directory not found: {args.data_dir}")
        sys.exit(1)
    
    try:
        # Initialize uploader
        uploader = VideoDataUploader(
            project_id=args.project_id,
            bucket_name=args.bucket_name,
            credentials_path=args.credentials
        )
        
        # Create bucket if requested
        if args.create_bucket:
            uploader.create_bucket_if_not_exists()
        
        # Upload dataset
        logger.info("🚀 Starting video dataset upload...")
        stats = uploader.upload_dataset(args.data_dir, args.gcs_prefix)
        
        # Print results
        print(f"\n📊 UPLOAD SUMMARY:")
        print(f"   ✅ Successful: {stats['success']}")
        print(f"   ⏭️ Skipped: {stats['skipped']}")
        print(f"   ❌ Failed: {stats['failed']}")
        print(f"   📍 GCS Location: gs://{args.bucket_name}/{args.gcs_prefix}")
        
        # Validate uploads if requested
        if args.validate:
            validation = uploader.validate_uploads(args.data_dir, args.gcs_prefix)
            print(f"\n🔍 VALIDATION RESULTS:")
            print(f"   ✅ Valid files: {validation['valid']}")
            print(f"   ❌ Missing files: {len(validation['missing'])}")
            print(f"   ⚠️ Size mismatches: {len(validation['size_mismatch'])}")
            
            if validation['missing']:
                print("   Missing files:", validation['missing'][:5])  # Show first 5
            if validation['size_mismatch']:
                print("   Size mismatches:", validation['size_mismatch'][:5])  # Show first 5
        
        # Create manifest if requested
        if args.create_manifest:
            uploader.create_dataset_manifest(args.gcs_prefix)
        
        logger.info("✅ Upload process completed!")
        
    except Exception as e:
        logger.error(f"❌ Upload failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()