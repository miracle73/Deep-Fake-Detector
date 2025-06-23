import { ZodError } from 'zod';

import { AppError } from '../utils/error.js';
import logger from '../utils/logger.js';

import type { Request, Response, NextFunction } from 'express';

export function errorHandler(
  err: unknown,
  req: Request,
  res: Response,
  next: NextFunction
) {
  if (err instanceof ZodError) {
    const formattedErrors = err.errors.map((e) => ({
      field: e.path.join('.') || 'unknown',
      message: e.message,
    }));
    return res.status(400).json({
      success: false,
      code: 400,
      message: 'Validation failed',
      errors: formattedErrors,
      ...(process.env.NODE_ENV !== 'production' && { stack: err.stack }),
    });
  }

  if (err instanceof AppError) {
    return res.status(err.statusCode).json(err);
  }

  logger.error('Unhandled error:', err);

  const message =
    process.env.NODE_ENV === 'production'
      ? 'Internal server error'
      : err instanceof Error
      ? err.message
      : 'Unknown error occurred';

  res.status(500).json({
    success: false,
    code: 500,
    message,
    ...(process.env.NODE_ENV !== 'production' && {
      error: err,
      stack: err instanceof Error ? err.stack : undefined,
    }),
  });
}
