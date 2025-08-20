#!/usr/bin/env python3
"""
Parallel preprocessing of videos from GCS - optimized for headless environments
Processes multiple videos simultaneously for 3x faster extraction
Works in GitHub Codespaces and other headless environments
"""

import os
import sys
import tempfile
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Optional
import time

# Configure environment for headless operation BEFORE importing any libraries
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '0'
os.environ['OPENCV_IO_ENABLE_JASPER'] = '0'
os.environ['OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS'] = '0'
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

# Import libraries
import numpy as np
from google.cloud import storage

# Import OpenCV with error handling for headless environments
try:
    import cv2
    # Test OpenCV functionality
    cv2.getVersionString()
except Exception as e:
    print(f"❌ OpenCV import error: {e}")
    print("📦 Installing opencv-python-headless for GitHub Codespaces...")
    os.system("pip install opencv-python-headless")
    try:
        import cv2
        print(f"✅ OpenCV headless installed successfully: {cv2.getVersionString()}")
    except Exception as e2:
        print(f"❌ Failed to install headless OpenCV: {e2}")
        sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HeadlessVideoPreprocessor:
    """Extract and preprocess frames from videos in headless environment"""
    
    def __init__(
        self,
        frame_rate: float = 1.0,
        frame_size: Tuple[int, int] = (224, 224),
        max_frames: int = 16,
        quality_threshold: float = 0.3
    ):
        self.frame_rate = frame_rate
        self.frame_size = frame_size
        self.max_frames = max_frames
        self.quality_threshold = quality_threshold
        
        # Verify OpenCV is working
        logger.info(f"🎥 OpenCV version: {cv2.getVersionString()} (headless mode)")
    
    def extract_frames(self, video_path: str) -> Optional[np.ndarray]:
        """Extract frames from video with quality filtering"""
        cap = None
        try:
            # Try multiple backends for better compatibility
            backends_to_try = [cv2.CAP_FFMPEG, cv2.CAP_ANY]
            
            for backend in backends_to_try:
                cap = cv2.VideoCapture(video_path, backend)
                if cap.isOpened():
                    break
                if cap is not None:
                    cap.release()
                    cap = None
                
            if cap is None or not cap.isOpened():
                logger.error(f"❌ Could not open video with any backend: {video_path}")
                return None
            
            # Get video properties with detailed logging
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = total_frames / fps if fps > 0 else 0
            
            logger.info(f"📹 Video info: {Path(video_path).name} - "
                       f"FPS: {fps:.2f}, Frames: {total_frames}, "
                       f"Size: {width}x{height}, Duration: {duration:.2f}s")
            
            # More lenient duration check
            if duration < 0.5:  # Reduced from 1.0 to 0.5 seconds
                logger.warning(f"⚠️ Video too short: {video_path} ({duration:.2f}s)")
                return None
            
            # Check for invalid video properties
            if fps <= 0 or total_frames <= 0:
                logger.warning(f"⚠️ Invalid video properties: FPS={fps}, Frames={total_frames}")
                # Try to read a few frames anyway
                test_frames = []
                for i in range(10):  # Try to read first 10 frames
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        test_frames.append(frame)
                    else:
                        break
                
                if len(test_frames) == 0:
                    logger.error(f"❌ Cannot read any frames from: {video_path}")
                    return None
                
                logger.info(f"📹 Read {len(test_frames)} test frames, proceeding with manual extraction")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
            
            # Calculate frame sampling - more flexible approach
            if fps > 0 and total_frames > 0:
                # Standard approach with known video properties
                frame_interval = max(1, int(fps / self.frame_rate))
                frames_to_extract = min(self.max_frames, int(duration * self.frame_rate))
                logger.info(f"🎯 Sampling every {frame_interval} frames, target: {frames_to_extract}")
            else:
                # Fallback for videos with unknown properties
                frame_interval = 30  # Sample every 30 frames
                frames_to_extract = self.max_frames
                logger.info(f"🎯 Using fallback sampling: every {frame_interval} frames")
            
            frames = []
            frame_count = 0
            consecutive_failures = 0
            max_consecutive_failures = 50
            
            while len(frames) < frames_to_extract and consecutive_failures < max_consecutive_failures:
                ret, frame = cap.read()
                
                if not ret or frame is None:
                    consecutive_failures += 1
                    if consecutive_failures > 10:
                        logger.warning(f"⚠️ {consecutive_failures} consecutive read failures")
                    continue
                
                consecutive_failures = 0  # Reset on successful read
                
                # Sample frames at specified interval
                if frame_count % frame_interval == 0:
                    # More lenient quality check
                    if frame.size > 0 and self._is_good_quality_frame(frame):
                        # Convert BGR to RGB
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        # Resize frame using INTER_LINEAR (good balance of speed/quality)
                        frame_resized = cv2.resize(
                            frame_rgb, 
                            self.frame_size, 
                            interpolation=cv2.INTER_LINEAR
                        )
                        frames.append(frame_resized)
                        
                        if len(frames) % 5 == 0:  # Log every 5 frames
                            logger.info(f"📸 Extracted {len(frames)}/{frames_to_extract} frames")
                
                frame_count += 1
                
                # Safety break for very long videos
                if frame_count > 10000:  # Prevent infinite loops
                    logger.warning(f"⚠️ Hit frame limit (10000), stopping extraction")
                    break
            
            if len(frames) == 0:
                logger.error(f"❌ No frames extracted from: {video_path}")
                logger.error(f"   Frame count reached: {frame_count}")
                logger.error(f"   Consecutive failures: {consecutive_failures}")
                return None
            
            logger.info(f"✅ Successfully extracted {len(frames)} frames from {Path(video_path).name}")
            
            # Pad to max_frames if needed (repeat last frame)
            if len(frames) < self.max_frames:
                last_frame = frames[-1]
                while len(frames) < self.max_frames:
                    frames.append(last_frame.copy())
            
            # Convert to numpy array: (frames, height, width, channels)
            frames_array = np.array(frames[:self.max_frames])
            return frames_array
            
        except Exception as e:
            logger.error(f"Error extracting frames from {video_path}: {e}")
            return None
        finally:
            if cap is not None:
                cap.release()
    
    def _is_good_quality_frame(self, frame: np.ndarray) -> bool:
        """Check if frame meets quality threshold - more lenient"""
        try:
            # Basic checks first
            if frame is None or frame.size == 0:
                return False
            
            # Check if frame is completely black/white
            mean_intensity = np.mean(frame)
            if mean_intensity < 5 or mean_intensity > 250:
                return False
            
            # More lenient blur detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate Laplacian variance (measure of blur)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Much more lenient threshold
            lenient_threshold = self.quality_threshold * 0.1  # 10x more lenient
            
            is_good = laplacian_var > lenient_threshold
            
            if not is_good:
                logger.debug(f"Frame quality check: {laplacian_var:.3f} < {lenient_threshold:.3f}")
            
            return is_good
            
        except Exception as e:
            logger.debug(f"Quality check error: {e}")
            return True  # If quality check fails, accept the frame

class ParallelGCSVideoPreprocessor:
    """Process videos from GCS in parallel for faster extraction"""
    
    def __init__(self, project_id: str, bucket_name: str, max_workers: int = 4):
        self.project_id = project_id
        self.bucket_name = bucket_name
        self.max_workers = max_workers
        
        # Initialize GCS client
        try:
            self.client = storage.Client(project=project_id)
            self.bucket = self.client.bucket(bucket_name)
            logger.info(f"✅ Connected to GCS bucket: {bucket_name}")
        except Exception as e:
            logger.error(f"❌ Failed to connect to GCS: {e}")
            raise
        
        # Statistics
        self.processed_count = 0
        self.failed_count = 0
        self.skipped_count = 0
        self.start_time = time.time()
    
    def create_preprocessor(self):
        """Create a preprocessor instance (one per thread)"""
        return HeadlessVideoPreprocessor(
            frame_rate=1.0,  # 1 frame per second
            frame_size=(224, 224),
            max_frames=16,  # Good balance for training
            quality_threshold=0.3
        )
    
    def process_bucket_videos(self):
        """Process all videos in parallel"""
        logger.info(f"🎬 Processing videos from gs://{self.bucket_name}")
        logger.info(f"🚀 Using {self.max_workers} parallel workers")
        logger.info(f"🖥️ Running in headless mode (GitHub Codespaces)")
        
        # Collect all video blobs
        all_videos = []
        
        logger.info("📋 Scanning bucket for videos...")
        for class_name in ['Real2', 'Fake2']:
            logger.info(f"   Scanning {class_name}/ directory...")
            
            blobs = self.client.list_blobs(self.bucket, prefix=f"{class_name}/")
            
            class_videos = []
            for blob in blobs:
                if blob.name.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    class_videos.append((blob, class_name))
            
            all_videos.extend(class_videos)
            logger.info(f"   Found {len(class_videos)} {class_name} videos")
        
        total_videos = len(all_videos)
        logger.info(f"📹 Total videos found: {total_videos}")
        
        if total_videos == 0:
            logger.warning("❌ No videos found in bucket!")
            return
        
        # Check for already processed videos
        logger.info("🔍 Checking for already processed videos...")
        processed_blobs = set()
        try:
            for blob in self.client.list_blobs(self.bucket, prefix="processed/"):
                if blob.name.endswith('.npz'):
                    processed_blobs.add(blob.name)
        except Exception as e:
            logger.warning(f"Could not check processed videos: {e}")
        
        videos_to_process = []
        for blob, class_name in all_videos:
            # Map class names: Real2 -> Real, Fake2 -> Fake
            output_class_name = class_name.replace('2', '')  # Remove the '2' suffix
            output_name = f"processed/{output_class_name}/{Path(blob.name).stem}.npz"
            if output_name not in processed_blobs:
                videos_to_process.append((blob, class_name))
            else:
                self.skipped_count += 1
                if self.skipped_count <= 5:  # Only show first 5
                    logger.info(f"⏭️ Skipping already processed: {blob.name}")
        
        if self.skipped_count > 5:
            logger.info(f"⏭️ ... and {self.skipped_count - 5} more already processed")
        
        logger.info(f"📊 Need to process: {len(videos_to_process)} videos")
        
        if not videos_to_process:
            logger.info("✅ All videos already processed!")
            return
        
        # Process videos in parallel
        logger.info(f"🚀 Starting parallel processing with {self.max_workers} workers...")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_video = {
                executor.submit(self.process_single_video_safe, blob, class_name): (blob.name, class_name)
                for blob, class_name in videos_to_process
            }
            
            # Process completed tasks
            for future in as_completed(future_to_video):
                video_name, class_name = future_to_video[future]
                
                try:
                    success = future.result()
                    if success:
                        self.processed_count += 1
                    else:
                        self.failed_count += 1
                        
                except Exception as e:
                    logger.error(f"❌ Error processing {video_name}: {e}")
                    self.failed_count += 1
                
                # Progress update every 10 videos
                total_done = self.processed_count + self.failed_count
                if total_done % 10 == 0 or total_done == len(videos_to_process):
                    elapsed = time.time() - self.start_time
                    rate = total_done / elapsed if elapsed > 0 else 0
                    eta = (len(videos_to_process) - total_done) / rate if rate > 0 else 0
                    
                    logger.info(f"📈 Progress: {total_done}/{len(videos_to_process)} "
                            f"({self.processed_count} ✅, {self.failed_count} ❌) "
                            f"Rate: {rate:.1f} videos/sec, ETA: {eta/60:.1f} min")
        
        # Final statistics
        elapsed = time.time() - self.start_time
        logger.info(f"\n{'='*60}")
        logger.info(f"🎉 Processing completed!")
        logger.info(f"📊 Final Statistics:")
        logger.info(f"   ✅ Successfully processed: {self.processed_count}")
        logger.info(f"   ⏭️ Already processed (skipped): {self.skipped_count}")
        logger.info(f"   ❌ Failed: {self.failed_count}")
        logger.info(f"   📈 Total videos in bucket: {total_videos}")
        logger.info(f"   ⏱️ Processing time: {elapsed/60:.1f} minutes")
        
        if self.processed_count > 0:
            logger.info(f"   🚀 Average speed: {self.processed_count/elapsed:.1f} videos/sec")
            logger.info(f"   📁 Results saved to: gs://{self.bucket_name}/processed/")
        
        if self.failed_count > 0:
            logger.warning(f"   ⚠️ {self.failed_count} videos failed - check logs above")
    
    def process_single_video_safe(self, blob, class_name: str) -> bool:
        """Process a single video with error handling"""
        video_name = Path(blob.name).stem
        tmp_video_path = None
        tmp_frames_path = None
        
        try:
            preprocessor = self.create_preprocessor()
            
            # Create temp file for video download
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_video:
                tmp_video_path = tmp_video.name
                
                # Download video from GCS
                blob.download_to_filename(tmp_video_path)
                
                # Extract frames
                frames = preprocessor.extract_frames(tmp_video_path)
                
                if frames is not None and len(frames) > 0:
                    # Map class names: Real2 -> Real, Fake2 -> Fake
                    output_class_name = class_name.replace('2', '')  # Remove the '2' suffix
                    
                    # Save processed frames back to GCS
                    output_blob_name = f"processed/{output_class_name}/{video_name}.npz"
                    
                    # Save to temp file first
                    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as tmp_frames:
                        tmp_frames_path = tmp_frames.name
                        np.savez_compressed(tmp_frames_path, frames=frames)
                        
                        # Upload to GCS
                        output_blob = self.bucket.blob(output_blob_name)
                        output_blob.upload_from_filename(tmp_frames_path)
                        
                        # Set metadata
                        output_blob.metadata = {
                            'processed_at': time.strftime('%Y-%m-%d %H:%M:%S UTC'),
                            'original_video': blob.name,
                            'num_frames': str(len(frames)),
                            'frame_shape': str(frames.shape),
                            'processing_environment': 'github_codespaces_headless'
                        }
                        output_blob.patch()
                    
                    logger.info(f"✅ {video_name}: {len(frames)} frames → {output_blob_name}")
                    return True
                else:
                    logger.warning(f"⚠️ {video_name}: No frames extracted")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ {video_name}: Failed - {str(e)[:100]}")
            return False
        finally:
            # Clean up temp files
            if tmp_video_path and os.path.exists(tmp_video_path):
                os.unlink(tmp_video_path)
            if tmp_frames_path and os.path.exists(tmp_frames_path):
                os.unlink(tmp_frames_path)

def test_single_video(project_id: str, bucket_name: str):
    """Test frame extraction on a single video for debugging"""
    print("🔍 Testing single video extraction...")
    
    try:
        client = storage.Client(project=project_id)
        bucket = client.bucket(bucket_name)
        
        # Find first video to test
        test_blob = None
        for class_name in ['Real2', 'Fake2']:
            blobs = client.list_blobs(bucket, prefix=f"{class_name}/", max_results=1)
            for blob in blobs:
                if blob.name.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    test_blob = blob
                    break
            if test_blob:
                break
        
        if not test_blob:
            print("❌ No test video found")
            return
        
        print(f"🎬 Testing with: {test_blob.name}")
        
        # Download and test
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_video:
            test_blob.download_to_filename(tmp_video.name)
            print(f"📥 Downloaded to: {tmp_video.name}")
            
            # Check file size
            file_size = os.path.getsize(tmp_video.name)
            print(f"📊 File size: {file_size / (1024*1024):.2f} MB")
            
            # Test extraction
            preprocessor = HeadlessVideoPreprocessor()
            frames = preprocessor.extract_frames(tmp_video.name)
            
            if frames is not None:
                print(f"✅ Success! Extracted {len(frames)} frames, shape: {frames.shape}")
            else:
                print("❌ Frame extraction failed")
            
            os.unlink(tmp_video.name)
            
    except Exception as e:
        print(f"❌ Test failed: {e}")

def main():
    """Main preprocessing function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Parallel preprocess videos in GCS (headless)')
    parser.add_argument('--project-id', type=str, required=True,
                       help='Google Cloud project ID')
    parser.add_argument('--bucket-name', type=str, 
                       default='second-bucket-video-deepfake',
                       help='GCS bucket name')
    parser.add_argument('--max-workers', type=int, default=6,
                       help='Number of parallel workers (default: 6)')
    parser.add_argument('--test', action='store_true',
                       help='Test single video extraction first')
    
    args = parser.parse_args()
    
    if args.test:
        test_single_video(args.project_id, args.bucket_name)
        return
    
    args = parser.parse_args()
    
    print("🎬 HEADLESS VIDEO DEEPFAKE PREPROCESSOR")
    print("=" * 55)
    print(f"🌐 Environment: GitHub Codespaces (Headless)")
    print(f"☁️ Project: {args.project_id}")
    print(f"🪣 Bucket: {args.bucket_name}")
    print(f"⚡ Workers: {args.max_workers}")
    print(f"🎥 OpenCV: {cv2.getVersionString()}")
    print("=" * 55)
    
    try:
        # Initialize preprocessor
        preprocessor = ParallelGCSVideoPreprocessor(
            project_id=args.project_id,
            bucket_name=args.bucket_name,
            max_workers=args.max_workers
        )
        
        # Process all videos
        start_time = time.time()
        preprocessor.process_bucket_videos()
        total_time = time.time() - start_time
        
        logger.info(f"\n🎉 Preprocessing pipeline completed!")
        logger.info(f"⏱️ Total execution time: {total_time/60:.1f} minutes")
        logger.info(f"📁 Processed frames available at: gs://{args.bucket_name}/processed/")
        
    except KeyboardInterrupt:
        logger.info("\n⏹️ Processing interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"\n❌ Preprocessing pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()