import { logInfo } from '../utils/google-cloud/logger.js';

import type { NextFunction, Request, Response } from 'express';

export const requestLogger = (
  req: Request,
  res: Response,
  next: NextFunction
) => {
  logInfo(`[${req.method}] ${req.originalUrl}`);

  next();
};
