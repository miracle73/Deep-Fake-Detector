#!/usr/bin/env python3
"""
Flask API for Deepfake Detection Model
Deploy this to Google Cloud Run
"""

from flask import Flask, request, jsonify
import torch
from PIL import Image
import io
import base64
import sys
import os
import logging
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import your custom modules
from models.efficientnet_detector import create_efficientnet_detector
from src.data.preprocessing import DeepfakePreprocessor
from config.model_configs import ConfigManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables
model = None
preprocessor = None
device = None

def load_model():
    """Load the trained deepfake detection model"""
    global model, preprocessor, device
    
    try:
        # Use CPU for Cloud Run (more cost-effective)
        device = torch.device('cpu')
        logger.info(f"Using device: {device}")
        
        # Load configuration (optional - can hardcode values)
        try:
            config_manager = ConfigManager('config/config.yaml')
            model_config = config_manager.model_config
            data_config = config_manager.data_config
        except:
            logger.warning("Config file not found, using default values")
            # Default values if config is missing
            model_config = type('obj', (object,), {
                'name': 'efficientnet-b4',
                'num_classes': 2,
                'pretrained': False,
                'dropout': 0.3
            })
            data_config = type('obj', (object,), {
                'input_size': 224,
                'augmentation': False
            })
        
        # Create model
        logger.info("Creating EfficientNet model...")
        model = create_efficientnet_detector(
            model_name=model_config.name,
            num_classes=model_config.num_classes,
            pretrained=model_config.pretrained,
            dropout=model_config.dropout
        )
        
        # Load trained weights
        checkpoint_path = 'checkpoints/best_model.pth'
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
        
        logger.info(f"Loading model weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load model state dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        model = model.to(device)
        
        # Create preprocessor
        logger.info("Setting up preprocessor...")
        preprocessor = DeepfakePreprocessor(
            input_size=data_config.input_size,
            normalize=True,
            augment=False  # No augmentation for inference
        )
        
        logger.info("✅ Model loaded successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        return False

@app.route('/', methods=['GET'])
def home():
    """Home endpoint with API information"""
    return jsonify({
        'service': 'Deepfake Detection API',
        'version': '1.0.0',
        'status': 'running',
        'endpoints': {
            'predict': '/predict (POST)',
            'health': '/health (GET)',
            'batch_predict': '/batch_predict (POST)'
        },
        'model_loaded': model is not None
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring"""
    return jsonify({
        'status': 'healthy' if model is not None else 'unhealthy',
        'model_loaded': model is not None,
        'device': str(device) if device else None
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
        processed_image = preprocessor.preprocess_image(image, augment=False)
        batch_input = processed_image.unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(batch_input)
            probabilities = torch.softmax(outputs, dim=1)
        
        # Extract results
        real_prob = probabilities[0, 0].item()
        fake_prob = probabilities[0, 1].item()
        
        # Determine prediction
        is_deepfake = fake_prob > 0.5
        confidence = max(real_prob, fake_prob)
        
        result = {
            'is_deepfake': is_deepfake,
            'deepfake_probability': round(fake_prob * 100, 2),  # e.g., 85.23%
            'real_probability': round(real_prob * 100, 2),      # e.g., 14.77%
            'confidence': round(confidence * 100, 2),           # e.g., 85.23%
            'predicted_class': 'deepfake' if is_deepfake else 'real',
            'threshold_used': 0.5
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
                processed_image = preprocessor.preprocess_image(image, augment=False)
                batch_input = processed_image.unsqueeze(0).to(device)
                
                with torch.no_grad():
                    outputs = model(batch_input)
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
            'successful_predictions': len([r for r in results if 'error' not in r])
        })
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({'error': f'Batch prediction failed: {str(e)}'}), 500

if __name__ == '__main__':
    # Load model when starting
    logger.info("Starting Deepfake Detection API...")
    success = load_model()
    
    if not success:
        logger.error("Failed to load model! Exiting...")
        exit(1)
    
    # Run the app
    port = int(os.environ.get('PORT', 8080))
    logger.info(f"Starting Flask app on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)

# For production deployment (gunicorn), load model at module level
else:
    logger.info("Loading model for production deployment...")
    load_model()