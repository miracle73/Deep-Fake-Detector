import logger from '../utils/logger.js';
import { cloudLogger } from '../utils/google-cloud/logger.js';

import type { NextFunction, Request, Response } from 'express';

export const requestLogger = async (
  req: Request,
  res: Response,
  next: NextFunction
) => {
  try {
    await cloudLogger.info({ message: `[${req.method}] ${req.originalUrl}` });
  } catch (error) {
    await cloudLogger.info({
      message: 'Failed to log request in requestLogger middleware',
      metadata: {
        error,
        method: req.method,
        url: req.originalUrl,
        headers: req.headers,
        ip: req.ip,
        timestamp: new Date().toISOString(),
      },
    });

    logger.error('Failed to log request:', error);
  }

  next();
};
