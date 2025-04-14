const express = require('express');
const { Storage } = require('@google-cloud/storage');
const multer = require('multer');
const path = require('path');
const fs = require('fs');

const app = express();
const port = process.env.PORT || 8080;

// GCP Configuration
const projectId = 'your-project-id'; // GCP project ID
const bucketName = `${projectId}-deepfake-user-media`;
const storage = new Storage({ projectId });

// Multer setup for file uploads
const upload = multer({
  dest: 'uploads/',
  limits: { fileSize: 10 * 1024 * 1024 }, // 10MB limit
  fileFilter: (req, file, cb) => {
    const allowedTypes = ['image/jpeg', 'image/png', 'video/mp4'];
    if (allowedTypes.includes(file.mimetype)) {
      cb(null, true);
    } else {
      cb(new Error('Unsupported file type'));
    }
  },
});

// Ensure uploads directory exists
const uploadDir = path.join(__dirname, 'uploads');
if (!fs.existsSync(uploadDir)) {
  fs.mkdirSync(uploadDir);
}

// Mock preprocessing function (to be expanded)
function preprocessMedia(filePath) {
  // Simulate preprocessing (e.g., frame extraction or resizing)
  return { status: 'processed', filename: path.basename(filePath) };
}

// Mock Vertex AI inference (to be replaced)
function detectDeepfake(processedData) {
  return { deepfakeProbability: 0.75, explanation: 'Mock detection result' };
}

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ status: 'healthy' });
});

// Deepfake detection endpoint
app.post('/detect', upload.single('media'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No media file provided' });
    }

    const filePath = req.file.path;

    // Preprocess media
    const processedData = preprocessMedia(filePath);
    if (!processedData) {
      throw new Error('Failed to process media');
    }

    // Upload to GCS
    const destination = `media/${req.file.originalname}`;
    await storage.bucket(bucketName).upload(filePath, {
      destination,
      metadata: { contentType: req.file.mimetype },
    });
    const gcsUrl = `gs://${bucketName}/${destination}`;

    // Mock detection (replace with Vertex AI call)
    const result = detectDeepfake(processedData);

    // Respond with result
    res.json({
      filename: req.file.originalname,
      mediaUrl: gcsUrl,
      deepfakeProbability: result.deepfakeProbability,
      explanation: result.explanation,
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  } finally {
    // Clean up local file
    if (fs.existsSync(filePath)) {
      fs.unlinkSync(filePath);
    }
  }
});

// Start server
app.listen(port, () => {
  console.log(`Server running on port ${port}`);
});