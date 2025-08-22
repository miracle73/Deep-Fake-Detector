#!/usr/bin/env python3
"""
Inference script for single audio file prediction
"""

import sys
import os
from pathlib import Path
import torch
import argparse
import numpy as np
import time
import json

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from audio_model import AudioDeepfakeDetector
from audio_preprocessing import AudioPreprocessor

class AudioInference:
    """Audio deepfake inference engine"""
    
    def __init__(self, model_path: str, device: str = None):
        """Initialize inference engine"""
        
        # Set device
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"🖥️ Using device: {self.device}")
        
        # Load model
        self.model = self._load_model(model_path)
        
        # Initialize preprocessor
        self.preprocessor = AudioPreprocessor(
            sample_rate=16000,
            n_mels=128,
            n_fft=2048,
            hop_length=512,
            max_duration=10.0,
            augment=False  # No augmentation for inference
        )
        
        print("✅ Inference engine ready!")
    
    def _load_model(self, model_path: str) -> AudioDeepfakeDetector:
        """Load trained model"""
        print(f"📁 Loading model from: {model_path}")
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Create model
        model = AudioDeepfakeDetector(
            input_size=128,
            hidden_size=256,
            num_layers=3,
            dropout=0.0,  # No dropout for inference
            bidirectional=True
        )
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        print(f"📊 Model loaded: {total_params:,} parameters")
        
        if 'val_acc' in checkpoint:
            print(f"📈 Model accuracy: {checkpoint['val_acc']:.2f}%")
        
        return model
    
    def predict_single(self, audio_path: str) -> dict:
        """Predict on a single audio file"""
        
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        print(f"\n🎵 Processing: {Path(audio_path).name}")
        
        # Start timing
        start_time = time.time()
        
        # Process audio
        mel_spec = self.preprocessor.process_audio_file(audio_path)
        
        if mel_spec is None:
            return {
                'error': 'Failed to process audio file',
                'audio_path': audio_path
            }
        
        # Convert to tensor
        mel_tensor = torch.from_numpy(mel_spec).float().unsqueeze(0).unsqueeze(0)
        mel_tensor = mel_tensor.to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(mel_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            
            fake_prob = probabilities[0, 1].item()
            real_prob = probabilities[0, 0].item()
            prediction = 1 if fake_prob > 0.5 else 0
            confidence = max(fake_prob, real_prob)
        
        # Calculate inference time
        inference_time = time.time() - start_time
        
        # Build result
        result = {
            'audio_path': audio_path,
            'prediction': 'FAKE' if prediction == 1 else 'REAL',
            'confidence': confidence,
            'fake_probability': fake_prob,
            'real_probability': real_prob,
            'inference_time_ms': inference_time * 1000,
            'status': 'success'
        }
        
        return result
    
    def predict_batch(self, audio_files: list) -> list:
        """Predict on multiple audio files"""
        results = []
        
        print(f"\n📊 Processing {len(audio_files)} audio files...")
        
        for audio_path in audio_files:
            try:
                result = self.predict_single(audio_path)
                results.append(result)
                
                # Print result
                print(f"  ✓ {Path(audio_path).name}: {result['prediction']} "
                      f"(confidence: {result['confidence']:.3f})")
                
            except Exception as e:
                print(f"  ✗ {Path(audio_path).name}: Error - {str(e)}")
                results.append({
                    'audio_path': audio_path,
                    'error': str(e),
                    'status': 'failed'
                })
        
        return results
    
    def predict_directory(self, directory: str, extensions: list = None) -> list:
        """Predict on all audio files in a directory"""
        
        if extensions is None:
            extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
        
        directory = Path(directory)
        if not directory.exists():
            raise ValueError(f"Directory not found: {directory}")
        
        # Find all audio files
        audio_files = []
        for ext in extensions:
            audio_files.extend(directory.rglob(f'*{ext}'))
        
        if not audio_files:
            print(f"⚠️ No audio files found in {directory}")
            return []
        
        print(f"🔍 Found {len(audio_files)} audio files")
        
        # Process all files
        results = self.predict_batch([str(f) for f in audio_files])
        
        # Calculate statistics
        total = len(results)
        successful = sum(1 for r in results if r.get('status') == 'success')
        fake_count = sum(1 for r in results if r.get('prediction') == 'FAKE')
        real_count = sum(1 for r in results if r.get('prediction') == 'REAL')
        
        # Summary
        summary = {
            'directory': str(directory),
            'total_files': total,
            'successful': successful,
            'failed': total - successful,
            'fake_detected': fake_count,
            'real_detected': real_count,
            'fake_percentage': (fake_count / successful * 100) if successful > 0 else 0
        }
        
        print(f"\n📈 Summary:")
        print(f"  Total: {total} files")
        print(f"  Successful: {successful}")
        print(f"  Fake: {fake_count} ({summary['fake_percentage']:.1f}%)")
        print(f"  Real: {real_count} ({100 - summary['fake_percentage']:.1f}%)")
        
        return results, summary

def main():
    parser = argparse.ArgumentParser(description='Audio Deepfake Detection Inference')
    parser.add_argument('--audio-file', type=str, help='Path to single audio file')
    parser.add_argument('--directory', type=str, help='Path to directory of audio files')
    parser.add_argument('--model-path', type=str, default='checkpoints/best_model.pth',
                       help='Path to trained model')
    parser.add_argument('--output', type=str, help='Output JSON file for results')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'],
                       help='Device to use for inference')
    
    args = parser.parse_args()
    
    if not args.audio_file and not args.directory:
        parser.error('Either --audio-file or --directory must be specified')
    
    # Initialize inference engine
    inference = AudioInference(args.model_path, args.device)
    
    # Perform inference
    if args.audio_file:
        # Single file prediction
        result = inference.predict_single(args.audio_file)
        
        # Print results
        print(f"\n{'='*50}")
        print(f"🎯 PREDICTION RESULT")
        print(f"{'='*50}")
        print(f"Audio: {Path(result['audio_path']).name}")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Fake Probability: {result['fake_probability']:.4f}")
        print(f"Real Probability: {result['real_probability']:.4f}")
        print(f"Inference Time: {result['inference_time_ms']:.2f}ms")
        
        results = [result]
        
    else:
        # Directory prediction
        results, summary = inference.predict_directory(args.directory)
    
    # Save results if output specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to: {args.output}")

if __name__ == "__main__":
    main()