
"""
Preprocess videos directly from GCS and save processed frames back to GCS
"""

import os
import sys
import tempfile
import logging
from pathlib import Path
from google.cloud import storage
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.video_preprocessing import VideoPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GCSVideoPreprocessor:
    """Process videos from GCS and save frames back to GCS"""
    
    def __init__(self, project_id: str, bucket_name: str):
        self.project_id = project_id
        self.bucket_name = bucket_name
        self.client = storage.Client(project=project_id)
        self.bucket = self.client.bucket(bucket_name)
        
        # Initialize video preprocessor
        self.preprocessor = VideoPreprocessor(
            frame_rate=1.0,  # 1 frame per second
            frame_size=(224, 224),
            max_frames=16,  # Reduced for CPU efficiency
            quality_threshold=0.3
        )
    
    def process_bucket_videos(self):
        """Process all videos in the bucket"""
        logger.info(f"🎬 Processing videos from gs://{self.bucket_name}")
        
        # Process both real and fake videos
        for class_name in ['real', 'fake']:
            logger.info(f"Processing {class_name} videos...")
            
            # List all videos in the class folder
            blobs = self.client.list_blobs(
                self.bucket, 
                prefix=f"{class_name}/"
            )
            
            video_count = 0
            for blob in blobs:
                if blob.name.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    try:
                        self.process_single_video(blob, class_name)
                        video_count += 1
                        
                        if video_count % 10 == 0:
                            logger.info(f"Processed {video_count} {class_name} videos...")
                            
                    except Exception as e:
                        logger.error(f"Failed to process {blob.name}: {e}")
            
            logger.info(f"✅ Processed {video_count} {class_name} videos")
    
    def process_single_video(self, blob, class_name: str):
        """Process a single video from GCS"""
        # Create temp file for video
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_video:
            # Download video to temp file
            blob.download_to_filename(tmp_video.name)
            
            # Extract frames
            frames = self.preprocessor.extract_frames(tmp_video.name)
            
            if frames is not None and len(frames) > 0:
                # Save processed frames back to GCS
                output_blob_name = f"processed/{class_name}/{Path(blob.name).stem}.npz"
                
                # Save to temp file first
                with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as tmp_frames:
                    np.savez_compressed(tmp_frames.name, frames=frames)
                    
                    # Upload to GCS
                    output_blob = self.bucket.blob(output_blob_name)
                    output_blob.upload_from_filename(tmp_frames.name)
                    
                    logger.info(f"✅ Saved frames to gs://{self.bucket_name}/{output_blob_name}")
                    
                    # Clean up temp file
                    os.unlink(tmp_frames.name)
            
            # Clean up temp video file
            os.unlink(tmp_video.name)

def main():
    """Main preprocessing function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess videos in GCS')
    parser.add_argument('--project-id', type=str, required=True,
                       help='Google Cloud project ID')
    parser.add_argument('--bucket-name', type=str, 
                       default='second-bucket-video-deepfake',
                       help='GCS bucket name')
    
    args = parser.parse_args()
    
    try:
        # Initialize preprocessor
        preprocessor = GCSVideoPreprocessor(
            project_id=args.project_id,
            bucket_name=args.bucket_name
        )
        
        # Process all videos
        preprocessor.process_bucket_videos()
        
        logger.info("✅ Preprocessing completed!")
        
    except Exception as e:
        logger.error(f"❌ Preprocessing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()