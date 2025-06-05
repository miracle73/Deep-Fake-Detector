import cors from 'cors';
import dotenv from 'dotenv';
import express from 'express';
import helmet from 'helmet';
import morgan from 'morgan';
import userRoutes from 'routes/userRoutes.js';

import swaggerJSDoc from 'swagger-jsdoc';

import swaggerUi from 'swagger-ui-express';

import connectDB from './config/db.js';
import { errorHandler } from './middlewares/error.js';
import { limiter } from './middlewares/rateLimit.js';
import authRoutes from './routes/authRoutes.js';
import billingRoutes from './routes/billingRoutes.js';
import { detectHandler } from './routes/detect.js';
import uploadRoutes from './routes/upload.js';
import logger from './utils/logger.js';
import { handleStripeWebhook } from './webhooks/stripeWebhookHandler.js';
import { swaggerOptions } from './config/swagger.js';

dotenv.config();

const app = express();
const port = process.env.PORT || 8080;

// connectDB();

app.use(helmet());
app.use(express.json());
app.use(cors());
app.use(express.urlencoded({ extended: true }));
app.use(morgan('dev'));
app.use(limiter);

const swaggerSpec = swaggerJSDoc(swaggerOptions);
-app.use('/api-docs', swaggerUi.serve, swaggerUi.setup(swaggerSpec));

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
app.use('/api/v1/user', userRoutes);
app.use('/api/v1/billing', billingRoutes);

app.post(
  '/webhook',
  express.raw({ type: 'application/json' }),
  handleStripeWebhook
);

app.use(errorHandler as express.ErrorRequestHandler);

app.listen(port, async () => {
  try {
    await connectDB();
    logger.info(`Server running🏃 on port ${port}...betta go catch it!🚀`);
    logger.info(
      `API Documentation available at http://localhost:${port}/api-docs`
    );
  } catch (error) {
    logger.error('Failed to start server:', error);
    process.exit(1);
  }
});

export default app;
