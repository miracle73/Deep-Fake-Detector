
#!/usr/bin/env python3
"""
Optimized inference script for deepfake detection
Supports single images, batch processing, and real-time inference
"""

import sys
import os
from pathlib import Path
import torch
import argparse
import time
import json
from typing import List, Dict, Union, Optional
import numpy as np
from PIL import Image
import cv2

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config.model_configs import ConfigManager
from models.efficientnet_detector import create_efficientnet_detector
from src.data.preprocessing import DeepfakePreprocessor
from src.utils.helpers import get_device, format_time, ensure_dir
from src.utils.logger import get_logger

class DeepfakeInference:
    """Optimized deepfake inference engine"""
    
    def __init__(
        self,
        checkpoint_path: str,
        config_path: str = "config/config.yaml",
        device: Optional[torch.device] = None,
        half_precision: bool = False
    ):
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.device = device or get_device()
        self.half_precision = half_precision and self.device.type == 'cuda'
        
        self.logger = get_logger("inference")
        
        # Load configuration
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.config
        
        # Load model
        self.model = self._load_model()
        
        # Create preprocessor
        self.preprocessor = DeepfakePreprocessor(
            input_size=self.config['data']['input_size'],
            normalize=True,
            augment=False
        )
        
        # Benchmark model
        self._benchmark_model()
        
        self.logger.info("🚀 DeepfakeInference engine ready!")
    
    def _load_model(self) -> torch.nn.Module:
        """Load and optimize model for inference"""
        self.logger.info(f"Loading model from {self.checkpoint_path}")
        
        # Create model
        model = create_efficientnet_detector(
            model_name=self.config['model']['name'],
            num_classes=self.config['model']['num_classes'],
            pretrained=False,  # We're loading weights
            dropout=self.config['model']['dropout']
        )
        
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Move to device
        model = model.to(self.device)
        
        # Set to evaluation mode
        model.eval()
        
        # Enable half precision if requested
        if self.half_precision:
            model = model.half()
            self.logger.info("✅ Half precision enabled")
        
        # Optimize for inference
        if hasattr(torch, 'jit') and self.device.type == 'cuda':
            # Create dummy input for tracing
            dummy_input = torch.randn(1, 3, self.config['data']['input_size'], 
                                    self.config['data']['input_size']).to(self.device)
            if self.half_precision:
                dummy_input = dummy_input.half()
            
            try:
                model = torch.jit.trace(model, dummy_input)
                self.logger.info("✅ Model optimized with TorchScript")
            except Exception as e:
                self.logger.warning(f"TorchScript optimization failed: {e}")
        
        self.logger.info("✅ Model loaded successfully")
        return model
    
    def _benchmark_model(self):
        """Benchmark model performance"""
        self.logger.info("🏁 Benchmarking model...")
        
        # Create test input
        test_input = torch.randn(1, 3, self.config['data']['input_size'], 
                               self.config['data']['input_size']).to(self.device)
        if self.half_precision:
            test_input = test_input.half()
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = self.model(test_input)
        
        # Benchmark
        torch.cuda.synchronize() if self.device.type == 'cuda' else None
        start_time = time.time()
        
        num_runs = 100
        with torch.no_grad():
            for _ in range(num_runs):
                _ = self.model(test_input)
        
        torch.cuda.synchronize() if self.device.type == 'cuda' else None
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs
        fps = 1.0 / avg_time
        
        self.logger.info(f"📊 Performance: {avg_time*1000:.2f}ms per image, {fps:.1f} FPS")
    
    def predict_single(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        return_probabilities: bool = True,
        return_features: bool = False
    ) -> Dict[str, Union[str, float, np.ndarray]]:
        """Predict on a single image"""
        
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image_path = str(image)
            image = Image.open(image).convert('RGB')
        else:
            image_path = "memory"
        
        # Preprocess image
        processed_image = self.preprocessor.preprocess_image(image, augment=False)
        
        # Add batch dimension and move to device
        batch_input = processed_image.unsqueeze(0).to(self.device)
        if self.half_precision:
            batch_input = batch_input.half()
        
        # Inference
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model(batch_input)
            probabilities = torch.softmax(outputs, dim=1)
            prediction = torch.argmax(outputs, dim=1)
        
        inference_time = time.time() - start_time
        
        # Extract results
        fake_prob = probabilities[0, 1].item()
        real_prob = probabilities[0, 0].item()
        predicted_class = "fake" if prediction[0].item() == 1 else "real"
        confidence = max(real_prob, fake_prob)
        
        # Build result dictionary
        result = {
            'image_path': image_path,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'inference_time_ms': inference_time * 1000
        }
        
        if return_probabilities:
            result.update({
                'real_probability': real_prob,
                'fake_probability': fake_prob,
                'probabilities': probabilities[0].cpu().numpy()
            })
        
        if return_features:
            # Extract features from the model
            features = self.model.get_feature_vector(batch_input)
            result['features'] = features[0].cpu().numpy()
        
        return result
    
    def predict_batch(
        self,
        images: List[Union[str, Path, Image.Image, np.ndarray]],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> List[Dict[str, Union[str, float]]]:
        """Predict on a batch of images"""
        results = []
        
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(range(0, len(images), batch_size), desc="Processing batches")
        else:
            iterator = range(0, len(images), batch_size)
        
        for i in iterator:
            batch_images = images[i:i + batch_size]
            batch_results = self._process_batch(batch_images)
            results.extend(batch_results)
        
        return results
    
    def _process_batch(self, images: List) -> List[Dict[str, Union[str, float]]]:
        """Process a single batch of images"""
        batch_tensors = []
        batch_paths = []
        
        # Preprocess all images in batch
        for image in images:
            if isinstance(image, (str, Path)):
                image_path = str(image)
                image = Image.open(image).convert('RGB')
            else:
                image_path = "memory"
            
            processed_image = self.preprocessor.preprocess_image(image, augment=False)
            batch_tensors.append(processed_image)
            batch_paths.append(image_path)
        
        # Stack into batch tensor
        batch_input = torch.stack(batch_tensors).to(self.device)
        if self.half_precision:
            batch_input = batch_input.half()
        
        # Inference
        with torch.no_grad():
            outputs = self.model(batch_input)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
        
        # Extract results
        results = []
        for i in range(len(batch_tensors)):
            fake_prob = probabilities[i, 1].item()
            real_prob = probabilities[i, 0].item()
            predicted_class = "fake" if predictions[i].item() == 1 else "real"
            confidence = max(real_prob, fake_prob)
            
            result = {
                'image_path': batch_paths[i],
                'predicted_class': predicted_class,
                'confidence': confidence,
                'real_probability': real_prob,
                'fake_probability': fake_prob
            }
            results.append(result)
        
        return results
    
    def predict_directory(
        self,
        directory: Union[str, Path],
        output_file: Optional[str] = None,
        batch_size: int = 32,
        extensions: List[str] = ['.jpg', '.jpeg', '.png']
    ) -> List[Dict[str, Union[str, float]]]:
        """Predict on all images in a directory"""
        directory = Path(directory)
        
        # Find all image files
        image_files = []
        for ext in extensions:
            image_files.extend(directory.rglob(f'*{ext}'))
            image_files.extend(directory.rglob(f'*{ext.upper()}'))
        
        self.logger.info(f"Found {len(image_files)} images in {directory}")
        
        if not image_files:
            self.logger.warning("No images found!")
            return []
        
        # Process all images
        results = self.predict_batch(image_files, batch_size=batch_size, show_progress=True)
        
        # Save results if output file specified
        if output_file:
            self._save_results(results, output_file)
        
        return results
    
    def _save_results(self, results: List[Dict], output_file: str):
        """Save results to file"""
        output_path = Path(output_file)
        ensure_dir(output_path.parent)
        
        # Add summary statistics
        total_images = len(results)
        fake_count = sum(1 for r in results if r['predicted_class'] == 'fake')
        real_count = total_images - fake_count
        avg_confidence = np.mean([r['confidence'] for r in results])
        
        summary = {
            'summary': {
                'total_images': total_images,
                'real_images': real_count,
                'fake_images': fake_count,
                'fake_percentage': (fake_count / total_images) * 100 if total_images > 0 else 0,
                'average_confidence': avg_confidence
            },
            'results': results
        }
        
        # Save as JSON
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info(f"💾 Results saved to {output_path}")

def main():
    """Main inference function"""
    parser = argparse.ArgumentParser(description='Deepfake Detection Inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--input', type=str, required=True,
                        help='Input image file or directory')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for results (JSON)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for processing')
    parser.add_argument('--half-precision', action='store_true',
                        help='Use half precision (FP16) for faster inference')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')
    
    args = parser.parse_args()
    
    print("🔍 DEEPFAKE DETECTION INFERENCE")
    print("=" * 50)
    
    # Check inputs
    if not Path(args.checkpoint).exists():
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        return
    
    if not Path(args.input).exists():
        print(f"❌ Input not found: {args.input}")
        return
    
    # Initialize inference engine
    try:
        inference_engine = DeepfakeInference(
            checkpoint_path=args.checkpoint,
            config_path=args.config,
            half_precision=args.half_precision
        )
    except Exception as e:
        print(f"❌ Failed to initialize inference engine: {e}")
        return
    
    # Determine input type
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single image
        print(f"🖼️ Processing single image: {input_path}")
        
        start_time = time.time()
        result = inference_engine.predict_single(input_path, return_probabilities=True)
        total_time = time.time() - start_time
        
        # Print results
        print(f"\n📊 Results:")
        print(f"   Predicted Class: {result['predicted_class'].upper()}")
        print(f"   Confidence: {result['confidence']:.4f}")
        print(f"   Real Probability: {result['real_probability']:.4f}")
        print(f"   Fake Probability: {result['fake_probability']:.4f}")
        print(f"   Inference Time: {result['inference_time_ms']:.2f}ms")
        print(f"   Total Time: {format_time(total_time)}")
        
        # Save single result if output specified
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            print(f"💾 Result saved to {args.output}")
    
    elif input_path.is_dir():
        # Directory of images
        print(f"📁 Processing directory: {input_path}")
        
        start_time = time.time()
        results = inference_engine.predict_directory(
            directory=input_path,
            output_file=args.output,
            batch_size=args.batch_size
        )
        total_time = time.time() - start_time
        
        # Print summary
        if results:
            total_images = len(results)
            fake_count = sum(1 for r in results if r['predicted_class'] == 'fake')
            real_count = total_images - fake_count
            avg_confidence = np.mean([r['confidence'] for r in results])
            
            print(f"\n📈 Summary:")
            print(f"   Total Images: {total_images}")
            print(f"   Real Images: {real_count} ({real_count/total_images*100:.1f}%)")
            print(f"   Fake Images: {fake_count} ({fake_count/total_images*100:.1f}%)")
            print(f"   Average Confidence: {avg_confidence:.4f}")
            print(f"   Total Time: {format_time(total_time)}")
            print(f"   Speed: {total_images/total_time:.1f} images/second")
            
            if args.verbose:
                print(f"\n📋 Detailed Results:")
                for result in results[:10]:  # Show first 10
                    print(f"   {Path(result['image_path']).name}: {result['predicted_class']} "
                          f"(conf: {result['confidence']:.3f})")
                if len(results) > 10:
                    print(f"   ... and {len(results) - 10} more")
    
    else:
        print(f"❌ Input must be a file or directory: {input_path}")

if __name__ == "__main__":
    main()
