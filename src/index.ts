import cors from 'cors';
import dotenv from 'dotenv';
import express from 'express';
import morgan from 'morgan';

import connectDB from './config/db.js';
import { errorHandler } from './middlewares/error.js';
import authRoutes from './routes/authRoutes.js';
import billingRoutes from './routes/billingRoutes.js';
import { detectHandler } from './routes/detect.js';
import uploadRoutes from './routes/upload.js';

import { handleStripeWebhook } from './webhooks/stripeWebhookHandler.js';

dotenv.config();

const app = express();
const port = process.env.PORT || 8080;

connectDB();

app.use(express.json());
app.use(cors());
app.use(express.urlencoded({ extended: true }));
app.use(morgan('dev'));

app.get('/', (req, res) => {
  res.status(200).json({
    message: 'Welcome to the image detection API',
    version: '1.0.0',
    endpoints: {
      '/detect': 'POST - Detect objects in an image',
      '/api/analyze': 'POST - Analyze an image',
      '/api/analyze/batch': 'POST - Analyze multiple images',
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
app.use('/api', uploadRoutes);

app.use('/api/v1/auth', authRoutes);
app.use('/api/v1/billing', billingRoutes);

app.post(
  '/webhook',
  express.raw({ type: 'application/json' }),
  handleStripeWebhook
);

app.use(errorHandler as express.ErrorRequestHandler);

app.listen(port, () => {
  console.log(`Server running🏃 on port ${port}...betta go catch it!🚀`);
});

export default app;
