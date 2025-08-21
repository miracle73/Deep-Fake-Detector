#!/usr/bin/env python3
"""
Video Deepfake Detection API - Main Entry Point with Keyframe Analysis
Handles video processing, keyframe extraction, and temporal analysis
"""

import os
import sys
import tempfile
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import json
import time

from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import numpy as np
import cv2

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.video_preprocessing import VideoPreprocessor
from src.video_model import VideoDeepfakeDetector

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables
model = None
device = None
model_info = {}
preprocessor = None

class KeyframeExtractor:
    """Extract keyframes from video using scene change detection"""
    
    def __init__(self, threshold: float = 30.0, min_interval: int = 30):
        """
        Args:
            threshold: Threshold for scene change detection (higher = more selective)
            min_interval: Minimum frames between keyframes
        """
        self.threshold = threshold
        self.min_interval = min_interval
    
    def extract_keyframes(self, video_path: str) -> List[Dict[str, Any]]:
        """
        Extract keyframes from video
        
        Returns:
            List of keyframe info with frame_number, timestamp, and confidence
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        keyframes = []
        prev_frame = None
        frame_idx = 0
        last_keyframe_idx = -self.min_interval
        
        # Always include first frame as keyframe
        ret, first_frame = cap.read()
        if ret:
            keyframes.append({
                'frame_number': 0,
                'timestamp': 0.0,
                'confidence': 100.0,
                'type': 'first_frame'
            })
            prev_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
            last_keyframe_idx = 0
            frame_idx = 1
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            timestamp = frame_idx / fps if fps > 0 else 0
            
            if prev_frame is not None and frame_idx - last_keyframe_idx >= self.min_interval:
                # Calculate frame difference
                diff = cv2.absdiff(gray_frame, prev_frame)
                mean_diff = np.mean(diff)
                
                if mean_diff > self.threshold:
                    keyframes.append({
                        'frame_number': frame_idx,
                        'timestamp': timestamp,
                        'confidence': min(100.0, mean_diff),
                        'type': 'scene_change'
                    })
                    last_keyframe_idx = frame_idx
            
            prev_frame = gray_frame
            frame_idx += 1
        
        # Always include last frame as keyframe if not already included
        if keyframes[-1]['frame_number'] != total_frames - 1:
            keyframes.append({
                'frame_number': total_frames - 1,
                'timestamp': duration,
                'confidence': 100.0,
                'type': 'last_frame'
            })
        
        # If we have too few keyframes, add evenly spaced ones
        if len(keyframes) < 3:
            additional_keyframes = []
            step = total_frames // 4
            for i in range(1, 4):
                frame_num = i * step
                if frame_num < total_frames and not any(abs(kf['frame_number'] - frame_num) < self.min_interval for kf in keyframes):
                    additional_keyframes.append({
                        'frame_number': frame_num,
                        'timestamp': frame_num / fps if fps > 0 else 0,
                        'confidence': 50.0,
                        'type': 'evenly_spaced'
                    })
            keyframes.extend(additional_keyframes)
            keyframes.sort(key=lambda x: x['frame_number'])
        
        cap.release()
        
        logger.info(f"Extracted {len(keyframes)} keyframes from video")
        return keyframes

class TemporalVideoPreprocessor:
    """Enhanced video preprocessor for keyframe-based analysis"""
    
    def __init__(self, frame_size: Tuple[int, int] = (224, 224), 
                 sequence_length: int = 16, quality_threshold: float = 0.3):
        self.frame_size = frame_size
        self.sequence_length = sequence_length
        self.quality_threshold = quality_threshold
        self.keyframe_extractor = KeyframeExtractor()
    
    def extract_keyframe_sequences(self, video_path: str) -> Dict[str, Any]:
        """
        Extract 16-frame sequences around each keyframe
        
        Returns:
            Dictionary with keyframes info and frame sequences
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        # Extract keyframes
        keyframes = self.keyframe_extractor.extract_keyframes(video_path)
        
        segments = []
        all_sequences = []
        
        for i, keyframe in enumerate(keyframes):
            keyframe_idx = keyframe['frame_number']
            
            # Define sequence range (8 frames before and after keyframe)
            start_frame = max(0, keyframe_idx - 8)
            end_frame = min(total_frames - 1, keyframe_idx + 8)
            
            # Ensure we have exactly 16 frames
            if end_frame - start_frame + 1 < self.sequence_length:
                # Adjust range to get 16 frames
                needed_frames = self.sequence_length - (end_frame - start_frame + 1)
                if start_frame == 0:
                    # Extend end
                    end_frame = min(total_frames - 1, end_frame + needed_frames)
                elif end_frame == total_frames - 1:
                    # Extend start
                    start_frame = max(0, start_frame - needed_frames)
                else:
                    # Extend both sides
                    extend_each = needed_frames // 2
                    start_frame = max(0, start_frame - extend_each)
                    end_frame = min(total_frames - 1, end_frame + extend_each)
            
            # Calculate temporal segment info
            start_time = start_frame / fps if fps > 0 else 0
            end_time = end_frame / fps if fps > 0 else duration
            
            segment_info = {
                'segment_id': i + 1,
                'keyframe_number': keyframe_idx,
                'keyframe_timestamp': keyframe['timestamp'],
                'keyframe_type': keyframe['type'],
                'sequence_start_frame': start_frame,
                'sequence_end_frame': end_frame,
                'sequence_start_time': start_time,
                'sequence_end_time': end_time,
                'duration': end_time - start_time,
                'frames_in_sequence': end_frame - start_frame + 1
            }
            
            # Extract frames for this sequence
            frames = []
            for frame_idx in range(start_frame, end_frame + 1):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    # Resize and normalize
                    frame = cv2.resize(frame, self.frame_size)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = frame.astype(np.float32) / 255.0
                    frames.append(frame)
            
            # Pad or trim to exactly sequence_length frames
            if len(frames) > self.sequence_length:
                frames = frames[:self.sequence_length]
            elif len(frames) < self.sequence_length:
                # Pad by repeating last frame
                while len(frames) < self.sequence_length:
                    frames.append(frames[-1] if frames else np.zeros((*self.frame_size, 3), dtype=np.float32))
            
            sequence_array = np.array(frames)  # Shape: (16, 224, 224, 3)
            
            segments.append(segment_info)
            all_sequences.append(sequence_array)
        
        cap.release()
        
        return {
            'segments': segments,
            'sequences': all_sequences,
            'video_info': {
                'total_frames': total_frames,
                'fps': fps,
                'duration': duration,
                'num_segments': len(segments)
            }
        }

class SafeguardMediaAnalyzer:
    """Analyzes predictions using SafeguardMedia confidence scoring"""
    
    @staticmethod
    def get_risk_assessment(real_confidence: float, fake_confidence: float) -> Dict[str, Any]:
        """
        Analyze prediction confidence using SafeguardMedia table
        
        Args:
            real_confidence: Confidence percentage for real prediction (0-100)
            fake_confidence: Confidence percentage for fake prediction (0-100)
            
        Returns:
            Dict with risk level and recommended action
        """
        
        if real_confidence >= 90 and fake_confidence <= 10:
            return {
                "interpretation": "Very Likely Real",
                "risk_level": "Low",
                "recommended_action": "Accept as authentic. Manual review optional.",
                "color_code": "green"
            }
        elif real_confidence >= 70 and fake_confidence <= 29:
            return {
                "interpretation": "Likely Real, Some Risk", 
                "risk_level": "Medium",
                "recommended_action": "Review manually if content is sensitive or high-stakes.",
                "color_code": "yellow"
            }
        elif 50 <= real_confidence <= 69 and 30 <= fake_confidence <= 49:
            return {
                "interpretation": "Ambiguous / Uncertain",
                "risk_level": "Medium-High", 
                "recommended_action": "Manual verification strongly recommended. Consider secondary tools.",
                "color_code": "orange"
            }
        elif 30 <= real_confidence <= 49 and 50 <= fake_confidence <= 69:
            return {
                "interpretation": "Likely Deepfake, But Not Conclusive",
                "risk_level": "High",
                "recommended_action": "Treat cautiously. Manual review required; possibly reject.",
                "color_code": "red"
            }
        elif real_confidence <= 29 and fake_confidence >= 70:
            return {
                "interpretation": "Very Likely Deepfake",
                "risk_level": "Very High",
                "recommended_action": "Reject or flag. Notify relevant stakeholders.",
                "color_code": "dark_red"
            }
        else:
            return {
                "interpretation": "Uncertain Classification",
                "risk_level": "Medium-High",
                "recommended_action": "Manual verification required.",
                "color_code": "orange"
            }

def load_model():
    """Load the trained video deepfake detection model"""
    global model, device, model_info, preprocessor
    
    try:
        # Use CPU (model was trained on CPU)
        device = torch.device('cpu')
        logger.info(f"Using device: {device}")
        
        # Model path - adjust this to your trained model location
        model_path = 'models/checkpoints/video_deepfake_detector.pth'
        
        # Alternative paths to try
        if not os.path.exists(model_path):
            alternative_paths = [
                'trained_models/video_deepfake_detector.pth',
                'models/experiments/video_deepfake_detector.pth',
                'checkpoints/video_deepfake_detector.pth'
            ]
            
            for alt_path in alternative_paths:
                if os.path.exists(alt_path):
                    model_path = alt_path
                    logger.info(f"Found model at: {alt_path}")
                    break
            else:
                # List available files to help debug
                for search_dir in ['models', 'trained_models', 'checkpoints']:
                    search_path = Path(search_dir)
                    if search_path.exists():
                        available_files = list(search_path.rglob('*.pth'))
                        logger.info(f"Available .pth files in {search_dir}: {available_files}")
                        if available_files:
                            model_path = str(available_files[0])
                            logger.info(f"Using first available model: {model_path}")
                            break
                else:
                    raise FileNotFoundError(f"No model file found. Please ensure your trained model exists.")
        
        logger.info(f"Loading model from: {model_path}")
        
        # Load model using the VideoDeepfakeDetector class
        model = VideoDeepfakeDetector.load_model(model_path)
        model = model.to(device)
        model.eval()
        
        # Extract model info if available
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict):
            if 'config' in checkpoint:
                model_info['training_config'] = checkpoint['config']
            if 'results' in checkpoint:
                model_info['training_results'] = checkpoint['results']
        
        # Initialize enhanced video preprocessor
        preprocessor = TemporalVideoPreprocessor(
            frame_size=(224, 224),
            sequence_length=16,
            quality_threshold=0.3
        )
        
        logger.info("Model and preprocessor loaded successfully!")
        logger.info(f"   Model type: VideoDeepfakeDetector")
        logger.info(f"   Sequence length: 16 frames")
        logger.info(f"   Frame size: (224, 224)")
        logger.info(f"   Analysis type: Keyframe-based temporal segments")
        
        return True
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False

@app.route('/', methods=['GET'])
def home():
    """Home endpoint with API information"""
    return jsonify({
        'service': 'Video Deepfake Detection API - Keyframe Analysis',
        'version': '2.0.0',
        'status': 'running',
        'model_type': 'VideoDeepfakeDetector (MobileNetV2 + LSTM)',
        'analysis_type': 'Keyframe-based Temporal Segments',
        'model_info': model_info,
        'endpoints': {
            'predict_video': '/predict_video (POST)',
            'analyze_temporal': '/analyze_temporal (POST)',
            'health': '/health (GET)',
            'model_info': '/model_info (GET)'
        },
        'supported_formats': ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'],
        'model_loaded': model is not None,
        'safeguard_media_integration': True,
        'keyframe_analysis': True
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy' if model is not None else 'unhealthy',
        'model_loaded': model is not None,
        'preprocessor_loaded': preprocessor is not None,
        'device': str(device) if device else None,
        'model_type': 'VideoDeepfakeDetector',
        'analysis_type': 'Keyframe-based Temporal Analysis'
    })

@app.route('/model_info', methods=['GET'])
def get_model_info():
    """Get detailed model information"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({
        'model_architecture': 'MobileNetV2 + LSTM',
        'task': 'Video Deepfake Detection',
        'analysis_method': 'Keyframe-based Temporal Segments',
        'input_requirements': {
            'format': 'Video file (mp4, avi, mov, mkv, wmv, flv)',
            'processing': 'Extracts keyframes and 16-frame sequences around each',
            'frame_size': '224x224',
            'sequence_length': '16 frames per segment',
            'keyframe_detection': 'Scene change detection + temporal sampling'
        },
        'output_format': {
            'temporal_segments': 'Per-segment analysis with time ranges',
            'segment_predictions': 'real or deepfake for each temporal segment',
            'confidence_scores': 'real_probability, deepfake_probability per segment',
            'safeguard_analysis': 'risk_level and recommended_action per segment',
            'overall_assessment': 'Aggregated video-level prediction'
        },
        'training_details': model_info,
        'safeguard_media_integration': True
    })

@app.route('/predict_video', methods=['POST'])
def predict_video():
    """Legacy endpoint - redirects to temporal analysis"""
    return analyze_temporal()

@app.route('/analyze_temporal', methods=['POST'])
def analyze_temporal():
    """
    Analyze video using keyframe-based temporal segments
    
    Accepts:
    - Form data with 'video' file
    - Supports: .mp4, .avi, .mov, .mkv, .wmv, .flv
    
    Returns:
    - JSON with per-segment predictions and overall assessment
    """
    if model is None or preprocessor is None:
        return jsonify({'error': 'Model or preprocessor not loaded'}), 500
    
    # Check if video file is provided
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided. Use form-data with "video" field'}), 400
    
    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'error': 'No video file selected'}), 400
    
    # Check file extension
    allowed_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'}
    file_ext = Path(video_file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        return jsonify({
            'error': f'Unsupported file format: {file_ext}. Supported: {list(allowed_extensions)}'
        }), 400
    
    temp_video_path = None
    
    try:
        start_time = time.time()
        
        # Save uploaded video to temporary file
        with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as temp_video:
            temp_video_path = temp_video.name
            video_file.save(temp_video_path)
        
        logger.info(f"Processing video with keyframe analysis: {video_file.filename}")
        
        # Extract keyframe sequences
        keyframe_data = preprocessor.extract_keyframe_sequences(temp_video_path)
        segments = keyframe_data['segments']
        sequences = keyframe_data['sequences']
        video_info = keyframe_data['video_info']
        
        if not sequences:
            return jsonify({
                'error': 'Could not extract keyframe sequences from video. The video may be too short or corrupted.'
            }), 400
        
        logger.info(f"Extracted {len(sequences)} temporal segments for analysis")
        
        # Analyze each temporal segment
        segment_results = []
        fake_votes = 0
        real_votes = 0
        
        for i, (segment_info, sequence) in enumerate(zip(segments, sequences)):
            # Make prediction for this segment
            prediction_result = model.predict(sequence)
            
            # Extract prediction details
            is_deepfake = prediction_result['is_fake']
            fake_probability = prediction_result['fake_prob']
            real_probability = prediction_result['real_prob']
            confidence = prediction_result['confidence']
            
            # Convert to percentages
            fake_percentage = round(fake_probability * 100, 2)
            real_percentage = round(real_probability * 100, 2)
            confidence_percentage = round(confidence * 100, 2)
            
            # Get SafeguardMedia risk assessment
            safeguard_analysis = SafeguardMediaAnalyzer.get_risk_assessment(
                real_percentage, fake_percentage
            )
            
            # Format time range
            start_time_str = f"{segment_info['sequence_start_time']:.1f}s"
            end_time_str = f"{segment_info['sequence_end_time']:.1f}s"
            time_range = f"{start_time_str}-{end_time_str}"
            
            segment_result = {
                'segment_id': segment_info['segment_id'],
                'time_range': time_range,
                'start_time': segment_info['sequence_start_time'],
                'end_time': segment_info['sequence_end_time'],
                'duration': segment_info['duration'],
                'keyframe_info': {
                    'frame_number': segment_info['keyframe_number'],
                    'timestamp': segment_info['keyframe_timestamp'],
                    'type': segment_info['keyframe_type']
                },
                'prediction': {
                    'is_deepfake': is_deepfake,
                    'predicted_class': 'deepfake' if is_deepfake else 'real',
                    'deepfake_probability': fake_percentage,
                    'real_probability': real_percentage,
                    'confidence': confidence_percentage
                },
                'safeguard_analysis': {
                    'interpretation': safeguard_analysis['interpretation'],
                    'risk_level': safeguard_analysis['risk_level'],
                    'recommended_action': safeguard_analysis['recommended_action'],
                    'color_code': safeguard_analysis['color_code']
                }
            }
            
            segment_results.append(segment_result)
            
            # Vote counting for overall assessment
            if is_deepfake:
                fake_votes += 1
            else:
                real_votes += 1
        
        # Calculate overall assessment
        total_segments = len(segment_results)
        fake_ratio = fake_votes / total_segments
        real_ratio = real_votes / total_segments
        
        overall_fake_percentage = round(fake_ratio * 100, 2)
        overall_real_percentage = round(real_ratio * 100, 2)
        
        # Overall classification based on majority vote
        overall_is_deepfake = fake_votes > real_votes
        overall_confidence = max(fake_ratio, real_ratio) * 100
        
        # Overall SafeguardMedia assessment
        overall_safeguard = SafeguardMediaAnalyzer.get_risk_assessment(
            overall_real_percentage, overall_fake_percentage
        )
        
        processing_time = time.time() - start_time
        
        # Create segment summary string
        segment_summary = []
        for result in segment_results:
            class_name = result['prediction']['predicted_class'].title()
            segment_summary.append(f"Segment {result['segment_id']} ({result['time_range']}): {class_name}")
        
        # Prepare comprehensive result
        result = {
            'video_filename': video_file.filename,
            'analysis_type': 'Keyframe-based Temporal Segments',
            'video_info': {
                'duration': f"{video_info['duration']:.1f}s",
                'total_frames': video_info['total_frames'],
                'fps': round(video_info['fps'], 2),
                'segments_analyzed': total_segments
            },
            'overall_assessment': {
                'is_deepfake': overall_is_deepfake,
                'predicted_class': 'deepfake' if overall_is_deepfake else 'real',
                'confidence': round(overall_confidence, 2),
                'fake_segments': fake_votes,
                'real_segments': real_votes,
                'fake_ratio': overall_fake_percentage,
                'real_ratio': overall_real_percentage,
                'safeguard_analysis': {
                    'interpretation': overall_safeguard['interpretation'],
                    'risk_level': overall_safeguard['risk_level'],
                    'recommended_action': overall_safeguard['recommended_action'],
                    'color_code': overall_safeguard['color_code']
                }
            },
            'segment_analysis': segment_results,
            'segment_summary': segment_summary,
            'technical_details': {
                'keyframe_extraction': 'Scene change detection + temporal sampling',
                'sequence_length': '16 frames per segment',
                'frame_size': '224x224',
                'processing_time_seconds': round(processing_time, 2),
                'model_type': 'VideoDeepfakeDetector',
                'total_frames_analyzed': total_segments * 16
            }
        }
        
        logger.info(f"Temporal analysis completed: {total_segments} segments analyzed")
        logger.info(f"Overall: {result['overall_assessment']['predicted_class']} "
                   f"({result['overall_assessment']['confidence']:.1f}% confidence)")
        logger.info(f"Risk Level: {overall_safeguard['risk_level']}")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Temporal analysis error: {str(e)}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500
        
    finally:
        # Clean up temporary files
        if temp_video_path and os.path.exists(temp_video_path):
            os.unlink(temp_video_path)

@app.route('/batch_analyze', methods=['POST'])
def batch_analyze_videos():
    """
    Analyze multiple videos with temporal segments
    Accepts multiple video files
    """
    if model is None or preprocessor is None:
        return jsonify({'error': 'Model or preprocessor not loaded'}), 500
    
    # Check if any files were uploaded
    if not request.files:
        return jsonify({'error': 'No video files provided'}), 400
    
    video_files = request.files.getlist('videos')
    if not video_files or all(f.filename == '' for f in video_files):
        return jsonify({'error': 'No video files selected'}), 400
    
    # Limit batch size
    if len(video_files) > 3:
        return jsonify({'error': 'Maximum 3 videos per batch for temporal analysis'}), 400
    
    results = []
    total_start_time = time.time()
    
    for i, video_file in enumerate(video_files):
        temp_video_path = None
        
        try:
            # Check file extension
            file_ext = Path(video_file.filename).suffix.lower()
            allowed_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'}
            
            if file_ext not in allowed_extensions:
                results.append({
                    'video_index': i,
                    'filename': video_file.filename,
                    'error': f'Unsupported format: {file_ext}'
                })
                continue
            
            # Process video
            with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as temp_video:
                temp_video_path = temp_video.name
                video_file.save(temp_video_path)
            
            # Extract keyframe sequences and analyze
            keyframe_data = preprocessor.extract_keyframe_sequences(temp_video_path)
            segments = keyframe_data['segments']
            sequences = keyframe_data['sequences']
            
            if not sequences:
                results.append({
                    'video_index': i,
                    'filename': video_file.filename,
                    'error': 'Could not extract keyframe sequences'
                })
                continue
            
            # Quick analysis for batch processing
            fake_votes = 0
            segment_count = len(sequences)
            
            for sequence in sequences:
                prediction_result = model.predict(sequence)
                if prediction_result['is_fake']:
                    fake_votes += 1
            
            real_votes = segment_count - fake_votes
            overall_is_deepfake = fake_votes > real_votes
            
            results.append({
                'video_index': i,
                'filename': video_file.filename,
                'overall_prediction': 'deepfake' if overall_is_deepfake else 'real',
                'segments_analyzed': segment_count,
                'fake_segments': fake_votes,
                'real_segments': real_votes,
                'confidence': round(max(fake_votes, real_votes) / segment_count * 100, 2)
            })
            
        except Exception as e:
            results.append({
                'video_index': i,
                'filename': video_file.filename,
                'error': f'Processing failed: {str(e)}'
            })
            
        finally:
            if temp_video_path and os.path.exists(temp_video_path):
                os.unlink(temp_video_path)
    
    total_processing_time = time.time() - total_start_time
    successful_predictions = len([r for r in results if 'error' not in r])
    
    return jsonify({
        'batch_results': results,
        'summary': {
            'total_videos': len(video_files),
            'successful_predictions': successful_predictions,
            'failed_predictions': len(video_files) - successful_predictions,
            'total_processing_time_seconds': round(total_processing_time, 2),
            'analysis_type': 'Keyframe-based Temporal Segments'
        }
    })

def main():
    """Main function to start the API server"""
    logger.info("Starting Video Deepfake Detection API - Keyframe Analysis...")
    logger.info("=" * 60)
    logger.info("Loading trained VideoDeepfakeDetector model...")
    
    # Load model and preprocessor
    success = load_model()
    
    if not success:
        logger.error("Failed to load model! Please ensure:")
        logger.error("1. Model file exists (video_deepfake_detector.pth)")
        logger.error("2. Model was trained using the training scripts")
        logger.error("3. PyTorch is installed")
        return
    
    logger.info("=" * 60)
    logger.info("API Features:")
    logger.info("- Keyframe-based temporal analysis")
    logger.info("- 16-frame sequences around each keyframe") 
    logger.info("- Scene change detection for keyframe extraction")
    logger.info("- Per-segment deepfake predictions")
    logger.info("- SafeguardMedia confidence analysis")
    logger.info("- Temporal segment visualization")
    logger.info("- Batch processing support")
    logger.info("- Multiple video format support")
    logger.info("=" * 60)
    
    # Start Flask app
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"API ready at http://localhost:{port}")
    logger.info("Endpoints:")
    logger.info(f"  POST /analyze_temporal - Keyframe-based temporal analysis")
    logger.info(f"  POST /predict_video - Legacy endpoint (redirects to temporal)")
    logger.info(f"  POST /batch_analyze - Batch temporal analysis") 
    logger.info(f"  GET  /health - Health check")
    logger.info(f"  GET  /model_info - Model information")
    logger.info("")
    logger.info("Example Response Format:")
    logger.info("  Segment 1 (0.0-3.2s): Real")
    logger.info("  Segment 2 (3.2-6.8s): Deepfake")
    logger.info("  Segment 3 (6.8-10.1s): Real")
    
    app.run(debug=False, host='0.0.0.0', port=port)

if __name__ == '__main__':
    main()