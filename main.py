#!/usr/bin/env python3
"""
Video Deepfake Detector - Main Entry Point
Handles video processing, training orchestration, and inference
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.video_preprocessing import VideoPreprocessor
from src.video_dataset import VideoDataset
from src.video_model import VideoDeepfakeDetector
from src.utils import setup_logging, load_config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_videos(data_dir: str, output_dir: str, config: dict):
    """Preprocess videos for training"""
    logger.info("🎬 Starting video preprocessing...")
    
    preprocessor = VideoPreprocessor(
        frame_rate=config.get('frame_rate', 1),  # Extract 1 frame per second
        frame_size=config.get('frame_size', (224, 224)),
        max_frames=config.get('max_frames', 30)  # Max 30 frames per video
    )
    
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    
    # Process videos in real and fake directories
    for class_name in ['real', 'fake']:
        class_dir = data_path / class_name
        if not class_dir.exists():
            logger.warning(f"Directory not found: {class_dir}")
            continue
            
        output_class_dir = output_path / class_name
        output_class_dir.mkdir(parents=True, exist_ok=True)
        
        video_files = list(class_dir.glob('*.mp4')) + list(class_dir.glob('*.avi'))
        logger.info(f"Processing {len(video_files)} {class_name} videos...")
        
        for video_file in video_files:
            try:
                frames = preprocessor.extract_frames(str(video_file))
                if frames:
                    # Save frames as a single file for this video
                    output_file = output_class_dir / f"{video_file.stem}.npz"
                    preprocessor.save_frames(frames, str(output_file))
                    logger.info(f"✅ Processed: {video_file.name}")
                else:
                    logger.warning(f"⚠️ No frames extracted from: {video_file.name}")
            except Exception as e:
                logger.error(f"❌ Error processing {video_file.name}: {e}")
    
    logger.info("✅ Video preprocessing completed!")

def train_model(data_dir: str, config: dict, use_vertex: bool = False):
    """Train the video deepfake detection model"""
    logger.info("🚀 Starting model training...")
    
    if use_vertex:
        logger.info("Using Vertex AI for training...")
        # This will be handled by vertex_ai_training_cpu_optimized.py
        os.system("python scripts/vertex_ai_training_cpu_optimized.py")
    else:
        logger.info("Training locally...")
        
        # Create dataset
        dataset = VideoDataset(
            data_dir=data_dir,
            split='train',
            config=config
        )
        
        # Create model
        model = VideoDeepfakeDetector(
            frame_encoder=config.get('frame_encoder', 'mobilenet_v2'),
            temporal_model=config.get('temporal_model', 'lstm'),
            num_frames=config.get('max_frames', 30)
        )
        
        # Train model (simplified for CPU)
        model.train_model(dataset, epochs=config.get('epochs', 10))
        
        # Save model
        model.save_model('models/checkpoints/video_deepfake_detector.pth')
        logger.info("✅ Model training completed!")

def predict_video(video_path: str, model_path: str, config: dict):
    """Predict if a video is deepfake or real"""
    logger.info(f"🔍 Analyzing video: {video_path}")
    
    # Load model
    model = VideoDeepfakeDetector.load_model(model_path)
    
    # Preprocess video
    preprocessor = VideoPreprocessor(
        frame_rate=config.get('frame_rate', 1),
        frame_size=config.get('frame_size', (224, 224)),
        max_frames=config.get('max_frames', 30)
    )
    
    frames = preprocessor.extract_frames(video_path)
    if not frames:
        logger.error("❌ Could not extract frames from video")
        return None
    
    # Make prediction
    prediction = model.predict(frames)
    
    result = {
        'video_path': video_path,
        'prediction': 'FAKE' if prediction['is_fake'] else 'REAL',
        'confidence': prediction['confidence'],
        'fake_probability': prediction['fake_prob'],
        'real_probability': prediction['real_prob']
    }
    
    logger.info(f"📊 Result: {result['prediction']} (confidence: {result['confidence']:.4f})")
    return result

def upload_data(data_dir: str):
    """Upload video data to Google Cloud Storage"""
    logger.info("☁️ Uploading data to GCS...")
    os.system(f"python scripts/upload_data_to_gcs.py --data-dir {data_dir}")

def setup_vertex_ai():
    """Setup Vertex AI environment"""
    logger.info("⚙️ Setting up Vertex AI...")
    os.system("bash scripts/setup_vertex_ai.sh")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Video Deepfake Detector')
    parser.add_argument('--mode', choices=['preprocess', 'train', 'predict', 'upload', 'setup'], 
                       required=True, help='Operation mode')
    parser.add_argument('--data-dir', type=str, default='data/raw',
                       help='Directory containing video data')
    parser.add_argument('--output-dir', type=str, default='data/processed',
                       help='Output directory for processed data')
    parser.add_argument('--video-path', type=str, help='Path to video for prediction')
    parser.add_argument('--model-path', type=str, default='models/checkpoints/video_deepfake_detector.pth',
                       help='Path to trained model')
    parser.add_argument('--config', type=str, default='config/training_config.yaml',
                       help='Configuration file')
    parser.add_argument('--use-vertex', action='store_true',
                       help='Use Vertex AI for training')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config) if Path(args.config).exists() else {
        'frame_rate': 1,
        'frame_size': [224, 224],
        'max_frames': 30,
        'epochs': 10,
        'frame_encoder': 'mobilenet_v2',
        'temporal_model': 'lstm'
    }
    
    try:
        if args.mode == 'setup':
            setup_vertex_ai()
            
        elif args.mode == 'upload':
            upload_data(args.data_dir)
            
        elif args.mode == 'preprocess':
            preprocess_videos(args.data_dir, args.output_dir, config)
            
        elif args.mode == 'train':
            train_model(args.output_dir, config, args.use_vertex)
            
        elif args.mode == 'predict':
            if not args.video_path:
                logger.error("❌ --video-path required for prediction mode")
                return
            
            if not Path(args.model_path).exists():
                logger.error(f"❌ Model not found: {args.model_path}")
                return
                
            result = predict_video(args.video_path, args.model_path, config)
            if result:
                print(f"\n🎯 PREDICTION RESULT:")
                print(f"   Video: {result['video_path']}")
                print(f"   Prediction: {result['prediction']}")
                print(f"   Confidence: {result['confidence']:.4f}")
                print(f"   Real Probability: {result['real_probability']:.4f}")
                print(f"   Fake Probability: {result['fake_probability']:.4f}")
        
        else:
            logger.error(f"❌ Unknown mode: {args.mode}")
            
    except Exception as e:
        logger.error(f"❌ Error in {args.mode} mode: {e}")
        sys.exit(1)

if __name__ == "__main__":
    print(" VIDEO DEEPFAKE DETECTOR")
    print("=" * 50)
    main()