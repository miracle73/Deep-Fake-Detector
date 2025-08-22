#!/usr/bin/env python3
"""
Flask Web Application for Audio Deepfake Detection
"""

import os
import sys
from pathlib import Path
import tempfile
import json
from datetime import datetime

from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import torch

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from audio_model import AudioDeepfakeDetector
from audio_preprocessing import AudioPreprocessor

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'wav', 'mp3', 'flac', 'm4a', 'ogg', 'aac'}

# Create upload directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static', exist_ok=True)

# Global model instance
model = None
preprocessor = None
device = None

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def load_model():
    """Load the trained model"""
    global model, preprocessor, device
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model_path = 'checkpoints/best_model.pth'
    
    if not Path(model_path).exists():
        print(f"⚠️ Model not found at {model_path}")
        print("Please train the model first using: python train.py")
        return False
    
    model = AudioDeepfakeDetector(
        input_size=128,
        hidden_size=256,
        num_layers=3,
        dropout=0.0,
        bidirectional=True
    )
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Initialize preprocessor
    preprocessor = AudioPreprocessor(
        sample_rate=16000,
        n_mels=128,
        n_fft=2048,
        hop_length=512,
        max_duration=10.0,
        augment=False
    )
    
    print(f"✅ Model loaded successfully on {device}")
    return True

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for audio deepfake detection"""
    
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    file = request.files['audio']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed: WAV, MP3, FLAC, M4A, OGG'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process audio
        mel_spec = preprocessor.process_audio_file(filepath)
        
        if mel_spec is None:
            return jsonify({'error': 'Failed to process audio file'}), 500
        
        # Prepare input
        mel_tensor = torch.from_numpy(mel_spec).float().unsqueeze(0).unsqueeze(0)
        mel_tensor = mel_tensor.to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(mel_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            
            fake_prob = probabilities[0, 1].item()
            real_prob = probabilities[0, 0].item()
            prediction = 'FAKE' if fake_prob > 0.5 else 'REAL'
            confidence = max(fake_prob, real_prob)
        
        # Clean up uploaded file
        os.remove(filepath)
        
        # Return result
        result = {
            'prediction': prediction,
            'confidence': round(confidence * 100, 2),
            'fake_probability': round(fake_prob * 100, 2),
            'real_probability': round(real_prob * 100, 2),
            'filename': file.filename,
            'timestamp': timestamp
        }
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device) if device else 'not initialized'
    })

# Create HTML template
def create_html_template():
    """Create index.html template"""
    html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Audio Deepfake Detector</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        
        .container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            padding: 40px;
            max-width: 600px;
            width: 100%;
        }
        
        h1 {
            text-align: center;
            color: #333;
            margin-bottom: 10px;
            font-size: 2.5em;
        }
        
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
        }
        
        .upload-area {
            border: 3px dashed #ddd;
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
            background: #fafafa;
        }
        
        .upload-area:hover {
            border-color: #667eea;
            background: #f0f0ff;
        }
        
        .upload-area.dragover {
            border-color: #667eea;
            background: #e0e0ff;
        }
        
        #file-input {
            display: none;
        }
        
        .upload-icon {
            font-size: 48px;
            color: #667eea;
            margin-bottom: 20px;
        }
        
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 15px 40px;
            border-radius: 50px;
            font-size: 16px;
            cursor: pointer;
            margin-top: 20px;
            transition: transform 0.2s;
            display: none;
        }
        
        .btn:hover {
            transform: scale(1.05);
        }
        
        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        
        .result {
            margin-top: 30px;
            padding: 20px;
            border-radius: 10px;
            display: none;
        }
        
        .result.real {
            background: #d4edda;
            border: 1px solid #c3e6cb;
        }
        
        .result.fake {
            background: #f8d7da;
            border: 1px solid #f5c6cb;
        }
        
        .result-title {
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        
        .result.real .result-title {
            color: #155724;
        }
        
        .result.fake .result-title {
            color: #721c24;
        }
        
        .result-details {
            color: #666;
        }
        
        .progress-bar {
            width: 100%;
            height: 30px;
            background: #f0f0f0;
            border-radius: 15px;
            overflow: hidden;
            margin-top: 10px;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            transition: width 0.3s;
        }
        
        .loading {
            display: none;
            text-align: center;
            margin-top: 20px;
        }
        
        .spinner {
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 20px auto;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .file-info {
            margin-top: 15px;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 5px;
            display: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎵 Audio Deepfake Detector</h1>
        <p class="subtitle">Upload an audio file to check if it's real or AI-generated</p>
        
        <div class="upload-area" id="upload-area">
            <div class="upload-icon">📁</div>
            <p>Click to select or drag & drop audio file here</p>
            <p style="margin-top: 10px; color: #999; font-size: 14px;">
                Supported: WAV, MP3, FLAC, M4A, OGG (Max 50MB)
            </p>
            <input type="file" id="file-input" accept="audio/*">
        </div>
        
        <div class="file-info" id="file-info"></div>
        
        <button class="btn" id="analyze-btn" onclick="analyzeAudio()">
            🔍 Analyze Audio
        </button>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>Analyzing audio...</p>
        </div>
        
        <div class="result" id="result">
            <div class="result-title" id="result-title"></div>
            <div class="result-details" id="result-details"></div>
            <div class="progress-bar">
                <div class="progress-fill" id="confidence-bar"></div>
            </div>
        </div>
    </div>
    
    <script>
        let selectedFile = null;
        
        const uploadArea = document.getElementById('upload-area');
        const fileInput = document.getElementById('file-input');
        const analyzeBtn = document.getElementById('analyze-btn');
        const fileInfo = document.getElementById('file-info');
        const loading = document.getElementById('loading');
        const result = document.getElementById('result');
        
        uploadArea.addEventListener('click', () => fileInput.click());
        
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                handleFileSelect(files[0]);
            }
        });
        
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFileSelect(e.target.files[0]);
            }
        });
        
        function handleFileSelect(file) {
            selectedFile = file;
            
            fileInfo.innerHTML = `
                <strong>Selected:</strong> ${file.name}<br>
                <strong>Size:</strong> ${(file.size / 1024 / 1024).toFixed(2)} MB
            `;
            fileInfo.style.display = 'block';
            analyzeBtn.style.display = 'inline-block';
            result.style.display = 'none';
        }
        
        async function analyzeAudio() {
            if (!selectedFile) return;
            
            analyzeBtn.disabled = true;
            loading.style.display = 'block';
            result.style.display = 'none';
            
            const formData = new FormData();
            formData.append('audio', selectedFile);
            
            try {
                const response = await fetch('/api/predict', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    showResult(data);
                } else {
                    alert('Error: ' + (data.error || 'Unknown error'));
                }
            } catch (error) {
                alert('Error: ' + error.message);
            } finally {
                analyzeBtn.disabled = false;
                loading.style.display = 'none';
            }
        }
        
        function showResult(data) {
            result.className = 'result ' + (data.prediction === 'REAL' ? 'real' : 'fake');
            result.style.display = 'block';
            
            document.getElementById('result-title').textContent = 
                data.prediction === 'REAL' ? '✅ REAL AUDIO' : '⚠️ FAKE AUDIO DETECTED';
            
            document.getElementById('result-details').innerHTML = `
                <strong>Confidence:</strong> ${data.confidence}%<br>
                <strong>Real Probability:</strong> ${data.real_probability}%<br>
                <strong>Fake Probability:</strong> ${data.fake_probability}%
            `;
            
            document.getElementById('confidence-bar').style.width = data.confidence + '%';
        }
    </script>
</body>
</html>'''
    
    # Create templates directory and save HTML
    os.makedirs('templates', exist_ok=True)
    with open('templates/index.html', 'w') as f:
        f.write(html_content)
    print("✅ HTML template created")

if __name__ == '__main__':
    # Create HTML template
    create_html_template()
    
    # Load model
    if load_model():
        print("🌐 Starting web server...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to load model. Exiting.")