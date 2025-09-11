from transformers import pipeline
from flask import Flask, request, jsonify
import os
from flask_cors import CORS
from werkzeug.utils import secure_filename
import tempfile
import requests

app = Flask(__name__)
CORS(app)

# Load the model once when server starts
print("Loading model...")
detector = pipeline("image-classification", model="Organika/sdxl-detector")
print("Model loaded successfully!")

# Configure upload settings
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/detect', methods=['POST'])
def detect_deepfake():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        try:
            # Run detection
            result = detector(filepath)
            
            # Clean up uploaded file
            os.remove(filepath)
            
            return jsonify({
                'label': result[0]['label'],
                'score': result[0]['score'],
                'is_deepfake': result[0]['label'] == 'artificial'
            })
        
        except Exception as e:
            # Clean up file if error occurs
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/detect-url', methods=['POST'])
def detect_deepfake_url():
    data = request.get_json()
    if not data or 'url' not in data:
        return jsonify({'error': 'No URL provided'}), 400
    
    url = data['url']
    
    # Check for unsupported URL patterns
    if 'instagram.com' in url or 'facebook.com' in url or 'twitter.com' in url:
        return jsonify({'error': 'Social media URLs are not supported. Please use a direct image link.'}), 400
    
    if 'drive.google.com' in url and '/file/d/' in url:
        return jsonify({'error': 'Google Drive URLs are not supported. Please use a direct image link.'}), 400
    
    try:
        response = requests.get(url, stream=True, timeout=10, allow_redirects=True)
        
        if response.status_code == 401:
            return jsonify({'error': 'URL requires authentication. Please use a public image link.'}), 400
        
        if response.status_code == 403:
            return jsonify({'error': 'Access forbidden. Please use a publicly accessible image link.'}), 400
        
        if response.status_code == 404:
            return jsonify({'error': 'Image not found. Please check the URL.'}), 400
        
        response.raise_for_status()
        
        content_type = response.headers.get('content-type', '')
        if not content_type.startswith('image/'):
            return jsonify({'error': 'URL does not point to an image file.'}), 400
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            for chunk in response.iter_content(chunk_size=8192):
                tmp_file.write(chunk)
            
            # Run detection
            result = detector(tmp_file.name)
            
            # Clean up
            os.unlink(tmp_file.name)
            
            return jsonify({
                'label': result[0]['label'],
                'score': result[0]['score'],
                'is_deepfake': result[0]['label'] == 'artificial'
            })
        
    except requests.exceptions.Timeout:
        return jsonify({'error': 'Request timed out. Please try a different URL.'}), 400
    except requests.exceptions.ConnectionError:
        return jsonify({'error': 'Unable to connect to the URL. Please check the link.'}), 400
    except Exception as e:
        return jsonify({'error': f'Failed to process URL: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

@app.route('/', methods=['GET'])
def home():
    return '''
    <h1>Deepfake Detection API</h1>
    <p>Upload an image to detect if it's artificial or human-generated.</p>
    <form action="/detect" method="post" enctype="multipart/form-data">
        <input type="file" name="file" accept="image/*" required>
        <button type="submit">Detect</button>
    </form>
    '''

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)