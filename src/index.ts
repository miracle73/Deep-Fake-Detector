import cors from 'cors';
import dotenv from 'dotenv';
import express from 'express';
import helmet from 'helmet';
import morgan from 'morgan';
import swaggerJSDoc from 'swagger-jsdoc';
import swaggerUi from 'swagger-ui-express';

import connectDB from './config/db.js';
import { startQueues } from './config/queues.js';
import { swaggerOptions } from './config/swagger.js';
import { errorHandler } from './middlewares/error.js';
import { limiter } from './middlewares/rateLimit.js';
import { requestLogger } from './middlewares/requestLogger.js';
import detectRoutes from './routes/analyze.js';
import authRoutes from './routes/authRoutes.js';
import businessIntegrationRoutes from './routes/businessIntegrationRoutes.js';
import demoRequestRoutes from './routes/demoRequestRoutes.js';
import { detectHandler } from './routes/detect.js';
import { feedbackRoutes } from './routes/feedback.js';
import notificationRoutes from './routes/notificationRoutes.js';
import subscriptionRoutes from './routes/subscriptionRoutes.js';
import userRoutes from './routes/userRoutes.js';
import waitlistRoutes from './routes/waitlistRoutes.js';
import { startQuotaResetSchedule } from './services/quotaReset.service.js';
import { cloudLogger } from './utils/google-cloud/logger.js';
import logger from './utils/logger.js';

import type { Request, Response } from 'express';

dotenv.config();

const app = express();
const port = process.env.PORT || 8080;

app.use(helmet());
app.use(
  '/api/v1/subscriptions/webhook',
  express.raw({ type: 'application/json' })
);
app.use(express.json());
app.use(cors());
app.use(express.urlencoded({ extended: true }));
app.use(morgan('dev'));
app.use(limiter);
app.use(requestLogger);

const swaggerSpec = swaggerJSDoc(swaggerOptions);
app.use('/api-docs', swaggerUi.serve, swaggerUi.setup(swaggerSpec));

app.get('/', async (req: Request, res: Response) => {
  res.status(200).json({
    message: 'Welcome to the image detection API',
    version: '1.0.0',
  });
});

app.get('/health', (req: Request, res: Response) => {
  res.status(200).json({
    status: 'ok',
    message: 'Server is running smoothly',
  });
});

app.post('/detect', detectHandler);
app.use('/api/v1/detect', detectRoutes);

app.use('/api/v1/auth', authRoutes);
app.use('/api/v1/user', userRoutes);
app.use('/api/v1/subscriptions', subscriptionRoutes);
app.use('/api/v1/waitlist', waitlistRoutes);
app.use('/api/v1/notifications', notificationRoutes);
app.use('/api/v1/feedback', feedbackRoutes);

// demo request endpoints
app.use('/api/v1/demo-request', demoRequestRoutes);
app.use('/api/v1/business-integration', businessIntegrationRoutes);

app.use(errorHandler as express.ErrorRequestHandler);

app.listen(port, async () => {
  try {
    await connectDB();

    startQuotaResetSchedule();

    startQueues();

    logger.info('Background queues and jobs initialized');

    logger.info(`Server running🏃 on port ${port}...betta go catch it!🚀`);
    logger.info(
      `API Documentation available at http://localhost:${port}/api-docs`
    );

    cloudLogger.info({
      message: 'This is a test log from the deepfake detector backend',
      context: { endpoint: '/health' },
      userId: 'system',
    });

    // cloudLogger.info({
    //   message: 'Detection event',
    //   context: { event_type: 'deepfake_detected' },
    //   userId: 'admin',
    // });
  } catch (error) {
    logger.error('Failed to start server:', error);
    process.exit(1);
  }
});

export default app;
