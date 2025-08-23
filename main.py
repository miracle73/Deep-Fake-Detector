#!/usr/bin/env python3
"""
Audio Deepfake Detection API - Simplified Response Format
Handles audio processing, temporal segment extraction, and deepfake detection
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
import librosa

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.audio_model import AudioDeepfakeDetector
from src.audio_preprocessing import AudioPreprocessor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables
model = None
device = None
model_info = {}
preprocessor = None

class TemporalSegmentExtractor:
    """Extract temporal segments from audio for analysis"""
    
    def __init__(self, segment_duration: float = 2.0, overlap: float = 0.5):
        """
        Args:
            segment_duration: Duration of each segment in seconds
            overlap: Overlap between segments (0.0 to 1.0)
        """
        self.segment_duration = segment_duration
        self.overlap = overlap
    
    def extract_segments(self, audio_path: str, sample_rate: int = 16000) -> List[Dict[str, Any]]:
        """
        Extract temporal segments from audio
        
        Returns:
            List of segment info with start_time, end_time, and segment_id
        """
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=sample_rate)
            duration = len(audio) / sr
            
            segments = []
            segment_samples = int(self.segment_duration * sr)
            hop_samples = int(segment_samples * (1 - self.overlap))
            
            segment_id = 1
            start_sample = 0
            
            while start_sample < len(audio):
                end_sample = min(start_sample + segment_samples, len(audio))
                start_time = start_sample / sr
                end_time = end_sample / sr
                
                segments.append({
                    'segment_id': segment_id,
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': end_time - start_time,
                    'start_sample': start_sample,
                    'end_sample': end_sample,
                    'type': 'temporal_segment'
                })
                
                segment_id += 1
                start_sample += hop_samples
                
                # Break if remaining audio is too short
                if end_sample >= len(audio):
                    break
            
            logger.info(f"Extracted {len(segments)} temporal segments from audio")
            return segments
            
        except Exception as e:
            logger.error(f"Error extracting segments: {e}")
            return []

class TemporalAudioPreprocessor:
    """Enhanced audio preprocessor for temporal segment analysis"""
    
    def __init__(self, sample_rate: int = 16000, n_mels: int = 128, 
                 n_fft: int = 2048, hop_length: int = 512,
                 segment_duration: float = 2.0, overlap: float = 0.5):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.segment_duration = segment_duration
        self.overlap = overlap
        self.segment_extractor = TemporalSegmentExtractor(segment_duration, overlap)
    
    def extract_segment_features(self, audio_path: str) -> Dict[str, Any]:
        """
        Extract mel-spectrogram features for each temporal segment
        
        Returns:
            Dictionary with segments info and feature arrays
        """
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            duration = len(audio) / sr
            
            # Extract segments
            segments = self.segment_extractor.extract_segments(audio_path, self.sample_rate)
            
            segment_features = []
            
            for segment in segments:
                start_sample = segment['start_sample']
                end_sample = segment['end_sample']
                
                # Extract audio segment
                audio_segment = audio[start_sample:end_sample]
                
                # Compute mel-spectrogram
                mel_spec = librosa.feature.melspectrogram(
                    y=audio_segment,
                    sr=self.sample_rate,
                    n_mels=self.n_mels,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length
                )
                
                # Convert to log scale
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                
                # Normalize
                mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)
                
                segment_features.append(mel_spec_db)
            
            return {
                'segments': segments,
                'features': segment_features,
                'audio_info': {
                    'total_duration': duration,
                    'sample_rate': self.sample_rate,
                    'num_segments': len(segments)
                }
            }
            
        except Exception as e:
            logger.error(f"Error extracting segment features: {e}")
            return None

def load_model():
    """Load the trained audio deepfake detection model"""
    global model, device, model_info, preprocessor
    
    try:
        # Use appropriate device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Model path - adjust this to your trained model location
        model_path = 'checkpoints/best_model.pth'
        
        # Alternative paths to try
        if not os.path.exists(model_path):
            alternative_paths = [
                'models/checkpoints/audio_deepfake_detector.pth',
                'trained_models/audio_deepfake_detector.pth',
                'models/experiments/audio_deepfake_detector.pth'
            ]
            
            for alt_path in alternative_paths:
                if os.path.exists(alt_path):
                    model_path = alt_path
                    logger.info(f"Found model at: {alt_path}")
                    break
            else:
                # List available files to help debug
                for search_dir in ['checkpoints', 'models', 'trained_models']:
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
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Create model with same parameters as training
        model = AudioDeepfakeDetector(
            input_size=128,
            hidden_size=256,
            num_layers=3,
            dropout=0.0,  # No dropout for inference
            bidirectional=True
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        # Extract model info if available
        if 'val_acc' in checkpoint:
            model_info['validation_accuracy'] = checkpoint['val_acc']
        if 'epoch' in checkpoint:
            model_info['trained_epochs'] = checkpoint['epoch']
        
        # Initialize enhanced audio preprocessor
        preprocessor = TemporalAudioPreprocessor(
            sample_rate=16000,
            n_mels=128,
            n_fft=2048,
            hop_length=512,
            segment_duration=2.0,  # 2-second segments
            overlap=0.5  # 50% overlap
        )
        
        logger.info("Model and preprocessor loaded successfully!")
        logger.info(f"   Model type: AudioDeepfakeDetector (CNN + BiLSTM)")
        logger.info(f"   Device: {device}")
        logger.info(f"   Segment duration: 2.0 seconds")
        logger.info(f"   Analysis type: Temporal segment-based")
        
        return True
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False

@app.route('/', methods=['GET'])
def home():
    """Home endpoint with API information"""
    return jsonify({
        'service': 'Audio Deepfake Detection API - Simplified',
        'version': '2.1.0',
        'status': 'running',
        'model_type': 'AudioDeepfakeDetector (CNN + BiLSTM)',
        'analysis_type': 'Temporal Segment-based',
        'model_info': model_info,
        'endpoints': {
            'predict_audio': '/predict_audio (POST)',
            'analyze_temporal': '/analyze_temporal (POST)',
            'health': '/health (GET)',
            'model_info': '/model_info (GET)'
        },
        'supported_formats': ['.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac'],
        'model_loaded': model is not None,
        'response_format': 'simplified'
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy' if model is not None else 'unhealthy',
        'model_loaded': model is not None,
        'preprocessor_loaded': preprocessor is not None,
        'device': str(device) if device else None,
        'model_type': 'AudioDeepfakeDetector',
        'analysis_type': 'Temporal Segment Analysis'
    })

@app.route('/model_info', methods=['GET'])
def get_model_info():
    """Get detailed model information"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return jsonify({
        'model_architecture': 'CNN + BiLSTM',
        'task': 'Audio Deepfake Detection',
        'analysis_method': 'Temporal Segments',
        'input_requirements': {
            'format': 'Audio file (wav, mp3, flac, m4a, ogg, aac)',
            'processing': 'Extracts 2-second segments with 50% overlap',
            'sample_rate': '16000 Hz',
            'features': 'Mel-spectrogram (128 mels)',
            'segment_duration': '2.0 seconds',
            'segment_overlap': '50%'
        },
        'output_format': 'simplified',
        'model_parameters': {
            'total': total_params,
            'trainable': trainable_params
        },
        'training_details': model_info
    })

@app.route('/predict_audio', methods=['POST'])
def predict_audio():
    """Main endpoint for audio deepfake detection - simplified response"""
    return analyze_temporal()

@app.route('/analyze_temporal', methods=['POST'])
def analyze_temporal():
    """
    Analyze audio using temporal segments - simplified response format
    
    Accepts:
    - Form data with 'audio' file
    - Supports: .wav, .mp3, .flac, .m4a, .ogg, .aac
    
    Returns:
    - Simplified JSON response
    """
    if model is None or preprocessor is None:
        return jsonify({'error': 'Model or preprocessor not loaded'}), 500
    
    # Check if audio file is provided
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided. Use form-data with "audio" field'}), 400
    
    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({'error': 'No audio file selected'}), 400
    
    # Check file extension
    allowed_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac'}
    file_ext = Path(audio_file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        return jsonify({
            'error': f'Unsupported file format: {file_ext}. Supported: {list(allowed_extensions)}'
        }), 400
    
    temp_audio_path = None
    
    try:
        start_time = time.time()
        
        # Save uploaded audio to temporary file
        with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as temp_audio:
            temp_audio_path = temp_audio.name
            audio_file.save(temp_audio_path)
        
        logger.info(f"Processing audio: {audio_file.filename}")
        
        # Extract temporal segment features
        segment_data = preprocessor.extract_segment_features(temp_audio_path)
        
        if segment_data is None:
            return jsonify({
                'error': 'Could not extract features from audio. The audio may be too short or corrupted.'
            }), 400
        
        segments = segment_data['segments']
        features = segment_data['features']
        audio_info = segment_data['audio_info']
        
        if not features:
            return jsonify({
                'error': 'Could not extract temporal segments from audio.'
            }), 400
        
        logger.info(f"Extracted {len(features)} temporal segments for analysis")
        
        # Analyze all segments and aggregate results
        total_fake_prob = 0.0
        total_real_prob = 0.0
        fake_votes = 0
        real_votes = 0
        
        for mel_spec in features:
            # Prepare input tensor
            mel_tensor = torch.from_numpy(mel_spec).float().unsqueeze(0).unsqueeze(0)
            mel_tensor = mel_tensor.to(device)
            
            # Make prediction
            with torch.no_grad():
                outputs = model(mel_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                
                fake_prob = probabilities[0, 1].item()
                real_prob = probabilities[0, 0].item()
                
                total_fake_prob += fake_prob
                total_real_prob += real_prob
                
                # Count votes
                if fake_prob > 0.5:
                    fake_votes += 1
                else:
                    real_votes += 1
        
        # Calculate overall results
        num_segments = len(features)
        avg_fake_prob = total_fake_prob / num_segments
        avg_real_prob = total_real_prob / num_segments
        
        # Final decision based on average probabilities
        is_deepfake = avg_fake_prob > avg_real_prob
        confidence = max(avg_fake_prob, avg_real_prob)
        
        # Convert to percentages and round
        fake_percentage = round(avg_fake_prob * 100, 2)
        real_percentage = round(avg_real_prob * 100, 2)
        confidence_percentage = round(confidence * 100, 2)
        
        # Create simplified response
        result = {
            'is_deepfake': is_deepfake,
            'deepfake_probability': fake_percentage,
            'real_probability': real_percentage,
            'confidence': confidence_percentage,
            'predicted_class': 'deepfake' if is_deepfake else 'real',
            'segments_processed': num_segments,
            'total_duration': round(audio_info['total_duration'], 1),
            'filename': audio_file.filename
        }
        
        processing_time = time.time() - start_time
        logger.info(f"Analysis completed in {processing_time:.2f}s: {result['predicted_class']} "
                   f"({result['confidence']:.1f}% confidence)")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500
        
    finally:
        # Clean up temporary files
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.unlink(temp_audio_path)

@app.route('/batch_analyze', methods=['POST'])
def batch_analyze_audio():
    """
    Analyze multiple audio files - simplified response format
    """
    if model is None or preprocessor is None:
        return jsonify({'error': 'Model or preprocessor not loaded'}), 500
    
    # Check if any files were uploaded
    if not request.files:
        return jsonify({'error': 'No audio files provided'}), 400
    
    audio_files = request.files.getlist('audios')
    if not audio_files or all(f.filename == '' for f in audio_files):
        return jsonify({'error': 'No audio files selected'}), 400
    
    # Limit batch size
    if len(audio_files) > 5:
        return jsonify({'error': 'Maximum 5 audio files per batch'}), 400
    
    results = []
    total_start_time = time.time()
    
    for i, audio_file in enumerate(audio_files):
        temp_audio_path = None
        
        try:
            # Check file extension
            file_ext = Path(audio_file.filename).suffix.lower()
            allowed_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac'}
            
            if file_ext not in allowed_extensions:
                results.append({
                    'filename': audio_file.filename,
                    'error': f'Unsupported format: {file_ext}'
                })
                continue
            
            # Process audio
            with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as temp_audio:
                temp_audio_path = temp_audio.name
                audio_file.save(temp_audio_path)
            
            # Extract segment features and analyze
            segment_data = preprocessor.extract_segment_features(temp_audio_path)
            
            if segment_data is None:
                results.append({
                    'filename': audio_file.filename,
                    'error': 'Could not extract features'
                })
                continue
            
            features = segment_data['features']
            audio_info = segment_data['audio_info']
            
            # Quick analysis
            total_fake_prob = 0.0
            total_real_prob = 0.0
            
            for mel_spec in features:
                mel_tensor = torch.from_numpy(mel_spec).float().unsqueeze(0).unsqueeze(0)
                mel_tensor = mel_tensor.to(device)
                
                with torch.no_grad():
                    outputs = model(mel_tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    
                    total_fake_prob += probabilities[0, 1].item()
                    total_real_prob += probabilities[0, 0].item()
            
            # Calculate averages
            num_segments = len(features)
            avg_fake_prob = total_fake_prob / num_segments
            avg_real_prob = total_real_prob / num_segments
            
            is_deepfake = avg_fake_prob > avg_real_prob
            confidence = max(avg_fake_prob, avg_real_prob)
            
            results.append({
                'is_deepfake': is_deepfake,
                'deepfake_probability': round(avg_fake_prob * 100, 2),
                'real_probability': round(avg_real_prob * 100, 2),
                'confidence': round(confidence * 100, 2),
                'predicted_class': 'deepfake' if is_deepfake else 'real',
                'segments_processed': num_segments,
                'total_duration': round(audio_info['total_duration'], 1),
                'filename': audio_file.filename
            })
            
        except Exception as e:
            results.append({
                'filename': audio_file.filename,
                'error': f'Processing failed: {str(e)}'
            })
            
        finally:
            if temp_audio_path and os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
    
    total_processing_time = time.time() - total_start_time
    successful_predictions = len([r for r in results if 'error' not in r])
    
    return jsonify({
        'results': results,
        'total_audios': len(audio_files),
        'successful_predictions': successful_predictions,
        'failed_predictions': len(audio_files) - successful_predictions,
        'total_processing_time_seconds': round(total_processing_time, 2)
    })

def main():
    """Main function to start the API server"""
    logger.info("Starting Audio Deepfake Detection API - Simplified Response Format...")
    logger.info("=" * 60)
    logger.info("Loading trained AudioDeepfakeDetector model...")
    
    # Load model and preprocessor
    success = load_model()
    
    if not success:
        logger.error("Failed to load model! Please ensure:")
        logger.error("1. Model file exists (best_model.pth or audio_deepfake_detector.pth)")
        logger.error("2. Model was trained using the training scripts")
        logger.error("3. PyTorch and librosa are installed")
        return
    
    logger.info("=" * 60)
    logger.info("API Features:")
    logger.info("- Temporal segment-based analysis")
    logger.info("- 2-second segments with 50% overlap") 
    logger.info("- Mel-spectrogram feature extraction")
    logger.info("- Simplified JSON response format")
    logger.info("- Multiple audio format support")
    logger.info("- Batch processing support")
    logger.info("=" * 60)
    
    # Start Flask app
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"API ready at http://localhost:{port}")
    logger.info("Endpoints:")
    logger.info(f"  POST /predict_audio - Main audio analysis endpoint")
    logger.info(f"  POST /analyze_temporal - Temporal segment analysis")
    logger.info(f"  POST /batch_analyze - Batch analysis") 
    logger.info(f"  GET  /health - Health check")
    logger.info(f"  GET  /model_info - Model information")
    logger.info("")
    logger.info("Response format:")
    logger.info('  {')
    logger.info('    "is_deepfake": false,')
    logger.info('    "deepfake_probability": 23.45,')
    logger.info('    "real_probability": 76.55,')
    logger.info('    "confidence": 76.55,')
    logger.info('    "predicted_class": "real",')
    logger.info('    "segments_processed": 3,')
    logger.info('    "total_duration": 12.3,')
    logger.info('    "filename": "audio.wav"')
    logger.info('  }')
    
    app.run(debug=False, host='0.0.0.0', port=port)

if __name__ == '__main__':
    main()