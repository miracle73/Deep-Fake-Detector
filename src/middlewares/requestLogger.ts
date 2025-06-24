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
    await logError('Failed to log request in requestLogger middleware', {
      error,
      method: req.method,
      url: req.originalUrl,
      headers: req.headers,
      ip: req.ip,
      timestamp: new Date().toISOString(),
    });
    console.error('Failed to log request:', error);
  }

  next();
};
