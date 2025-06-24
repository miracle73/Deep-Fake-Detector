import { logError, logInfo } from '../utils/google-cloud/logger.js';

import type { NextFunction, Request, Response } from 'express';

export const requestLogger = async (
  req: Request,
  res: Response,
  next: NextFunction
) => {
  try {
    await logInfo(`[${req.method}] ${req.originalUrl}`);
  } catch (error) {
    await logError(`[${req.method}] ${req.originalUrl}`, '');
    console.error('Failed to log request:', error);
  }

  next();
};
