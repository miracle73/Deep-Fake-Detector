#!/usr/bin/env python3
"""
Flask API for Deepfake Detection Model
Updated to use the CPU-trained MobileNetV2 model
Optimized for Google App Engine deployment
"""

from flask import Flask, request, jsonify
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import base64
import os
import logging
import json
from pathlib import Path

# App Engine specific configuration
if os.environ.get('GAE_ENV', '').startswith('standard'):
    # Running on App Engine
    import logging
    logging.basicConfig(level=logging.INFO)
else:
    # Local development
    logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables
model = None
transform = None
device = None
model_info = {}

class MobileNetDetector(nn.Module):
    """MobileNetV2 - matching the training script architecture"""
    
    def __init__(self, num_classes=2, dropout=0.2):
        super().__init__()
        
        # Use MobileNetV2 - same as training
        self.base_model = models.mobilenet_v2(pretrained=False)
        
        # Get the number of input features for the classifier
        num_features = self.base_model.classifier[1].in_features
        
        # Replace classifier for binary classification
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(num_features, num_classes)
        )
    
    def forward(self, x):
        return self.base_model(x)

def load_model():
    """Load the trained deepfake detection model"""
    global model, transform, device, model_info
    
    try:
        # Use CPU (since model was trained on CPU)
        device = torch.device('cpu')
        logger.info(f"Using device: {device}")
        
        # Define the exact path to your downloaded model
        checkpoint_path = 'checkpoints/new/best_model_cpu_30000_samples.pth'
        
        # Alternative paths to try
        if not os.path.exists(checkpoint_path):
            alternative_paths = [
                'checkpoints/new/best_model_cpu_10000_samples.pth',
                'checkpoints/new/best_model_cpu_20000_samples.pth',
                'checkpoints/deepfake-cpu-optimized-20250809_222222/best_model_cpu_30000_samples.pth',
                'checkpoints/new/best_model.pth'
            ]
            
            for alt_path in alternative_paths:
                if os.path.exists(alt_path):
                    checkpoint_path = alt_path
                    logger.info(f"Found model at: {alt_path}")
                    break
            else:
                # List available files to help debug
                checkpoints_dir = Path('checkpoints')
                if checkpoints_dir.exists():
                    available_files = list(checkpoints_dir.rglob('*.pth'))
                    logger.error(f"Available .pth files: {available_files}")
                raise FileNotFoundError(f"Model checkpoint not found. Looked for: {checkpoint_path}")
        
        logger.info(f"Loading model weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Extract model info from checkpoint
        if 'config' in checkpoint:
            model_info['training_config'] = checkpoint['config']
        if 'val_acc' in checkpoint:
            model_info['validation_accuracy'] = checkpoint['val_acc']
        if 'train_acc' in checkpoint:
            model_info['training_accuracy'] = checkpoint['train_acc']
        if 'sample_size' in checkpoint:
            model_info['trained_on_samples'] = checkpoint['sample_size']
        if 'epoch' in checkpoint:
            model_info['trained_epochs'] = checkpoint['epoch']
        
        logger.info(f"Model info: {model_info}")
        
        # Create model with same architecture as training
        model = MobileNetDetector(num_classes=2, dropout=0.2)
        
        # Load model state dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        model = model.to(device)
        
        # Create transform - same as used in training
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load training summary if available
        summary_path = checkpoint_path.replace('.pth', '.json').replace('best_model_cpu', 'training_summary')
        if not os.path.exists(summary_path):
            summary_path = 'checkpoints/new/training_summary.json'
        
        if os.path.exists(summary_path):
            with open(summary_path, 'r') as f:
                summary = json.load(f)
                model_info['training_summary'] = summary
                logger.info(f"Loaded training summary: {summary}")
        
        logger.info("✅ Model loaded successfully!")
        logger.info(f"   Model type: MobileNetV2")
        logger.info(f"   Trained on: {model_info.get('trained_on_samples', 'unknown')} samples")
        logger.info(f"   Validation accuracy: {model_info.get('validation_accuracy', 'unknown')}%")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        return False

@app.route('/', methods=['GET'])
def home():
    """Home endpoint with API information"""
    return jsonify({
        'service': 'Deepfake Detection API',
        'version': '2.0.0',
        'status': 'running',
        'model_type': 'MobileNetV2 (CPU-optimized)',
        'model_info': model_info,
        'endpoints': {
            'predict': '/predict (POST)',
            'health': '/health (GET)',
            'batch_predict': '/batch_predict (POST)',
            'model_info': '/model_info (GET)'
        },
        'model_loaded': model is not None,
        'environment': 'App Engine' if os.environ.get('GAE_ENV') else 'Local'
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring"""
    return jsonify({
        'status': 'healthy' if model is not None else 'unhealthy',
        'model_loaded': model is not None,
        'device': str(device) if device else None,
        'model_type': 'MobileNetV2',
        'environment': 'App Engine' if os.environ.get('GAE_ENV') else 'Local'
    })

@app.route('/model_info', methods=['GET'])
def get_model_info():
    """Get detailed information about the loaded model"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({
        'model_architecture': 'MobileNetV2',
        'trained_for_task': 'Deepfake Detection',
        'input_size': '224x224',
        'normalization': {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        },
        'training_details': model_info,
        'notes': 'This model was trained on a sampled subset for CPU efficiency',
        'environment': 'App Engine' if os.environ.get('GAE_ENV') else 'Local'
    })

@app.route('/predict', methods=['POST'])
def predict_image():
    """
    Predict if a single image is a deepfake
    
    Accepts:
    - Form data with 'image' file
    - JSON with 'image_base64' field
    
    Returns:
    - JSON with prediction results and percentages
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Get image from request
        image = None
        
        if 'image' in request.files:
            # File upload method
            image_file = request.files['image']
            if image_file.filename == '':
                return jsonify({'error': 'No image file selected'}), 400
            image = Image.open(image_file.stream).convert('RGB')
            
        elif request.is_json and 'image_base64' in request.json:
            # Base64 encoded image method
            try:
                image_data = base64.b64decode(request.json['image_base64'])
                image = Image.open(io.BytesIO(image_data)).convert('RGB')
            except Exception as e:
                return jsonify({'error': f'Invalid base64 image: {str(e)}'}), 400
                
        else:
            return jsonify({
                'error': 'No image provided. Use form-data with "image" field or JSON with "image_base64" field'
            }), 400
        
        # Preprocess image
        logger.info(f"Processing image of size: {image.size}")
        processed_image = transform(image).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(processed_image)
            probabilities = torch.softmax(outputs, dim=1)
        
        # Extract results
        real_prob = probabilities[0, 0].item()
        fake_prob = probabilities[0, 1].item()
        
        # Determine prediction
        is_deepfake = fake_prob > 0.5
        confidence = max(real_prob, fake_prob)
        
        result = {
            'is_deepfake': is_deepfake,
            'deepfake_probability': round(fake_prob * 100, 2),
            'real_probability': round(real_prob * 100, 2),
            'confidence': round(confidence * 100, 2),
            'predicted_class': 'deepfake' if is_deepfake else 'real',
            'threshold_used': 0.5,
            'model_type': 'MobileNetV2',
            'trained_samples': model_info.get('trained_on_samples', 'unknown')
        }
        
        logger.info(f"Prediction result: {result['predicted_class']} ({result['confidence']}% confidence)")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """
    Predict multiple images at once
    Expects JSON with array of base64 images
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        if not request.is_json or 'images' not in request.json:
            return jsonify({'error': 'JSON with "images" array required'}), 400
        
        images_data = request.json['images']
        if not isinstance(images_data, list):
            return jsonify({'error': '"images" must be an array'}), 400
        
        if len(images_data) > 10:  # Limit batch size
            return jsonify({'error': 'Maximum 10 images per batch'}), 400
        
        results = []
        
        for i, image_base64 in enumerate(images_data):
            try:
                # Decode image
                image_data = base64.b64decode(image_base64)
                image = Image.open(io.BytesIO(image_data)).convert('RGB')
                
                # Process and predict
                processed_image = transform(image).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    outputs = model(processed_image)
                    probabilities = torch.softmax(outputs, dim=1)
                
                real_prob = probabilities[0, 0].item()
                fake_prob = probabilities[0, 1].item()
                is_deepfake = fake_prob > 0.5
                confidence = max(real_prob, fake_prob)
                
                results.append({
                    'image_index': i,
                    'is_deepfake': is_deepfake,
                    'deepfake_probability': round(fake_prob * 100, 2),
                    'real_probability': round(real_prob * 100, 2),
                    'confidence': round(confidence * 100, 2),
                    'predicted_class': 'deepfake' if is_deepfake else 'real'
                })
                
            except Exception as e:
                results.append({
                    'image_index': i,
                    'error': f'Failed to process image {i}: {str(e)}'
                })
        
        return jsonify({
            'batch_results': results,
            'total_images': len(images_data),
            'successful_predictions': len([r for r in results if 'error' not in r]),
            'model_type': 'MobileNetV2'
        })
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({'error': f'Batch prediction failed: {str(e)}'}), 500

# Load model when the module is imported (for production)
if __name__ != '__main__':
    logger.info("Loading model for production deployment...")
    load_model()

if __name__ == '__main__':
    # Load model when starting locally
    logger.info("Starting Deepfake Detection API...")
    logger.info("=" * 60)
    logger.info("Using CPU-trained MobileNetV2 model")
    logger.info("=" * 60)
    
    success = load_model()
    
    if not success:
        logger.error("Failed to load model! Please check:")
        logger.error("1. Model file exists in checkpoints/new/")
        logger.error("2. File name matches: best_model_cpu_30000_samples.pth")
        logger.error("3. PyTorch is installed: pip install torch torchvision")
        exit(1)
    
    # For local development
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting Flask app on port {port}")
    logger.info(f"API ready at http://localhost:{port}")
    app.run(debug=True, host='0.0.0.0', port=port)

# App Engine will automatically handle the WSGI server, so the above is only for local testing