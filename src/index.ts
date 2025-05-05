import cors from 'cors';
import dotenv from 'dotenv';
import express from 'express';

import { detectHandler } from './routes/detect.js';
import uploadRoutes from './routes/upload.js';

dotenv.config({ path: `.env.${process.env.ENV || 'development'}` });

const app = express();
const port = process.env.PORT || 8080;

app.use(express.json());
app.use(cors());

app.get('/', (req, res) => {
  res.status(200).json({
    message: 'Welcome to the image detection API',
    version: '1.0.0',
    endpoints: {
      '/detect': 'POST - Detect objects in an image',
      '/media/upload': 'POST - Upload an image for detection',
      '/media/upload/batch': 'POST - Upload multiple images for detection',
    },
  });
});

app.get('/health', (req, res) => {
  res.status(200).json({
    status: 'ok',
    message: 'Server is running smoothly',
  });
});

app.post('/detect', detectHandler);
app.use('media', uploadRoutes);

app.listen(port, () => {
  console.log(`Server running🏃 on port ${port}...betta go catch it!🚀`);
});
