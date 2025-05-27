import type { Request, Response, NextFunction } from 'express';
import { ZodError } from 'zod';

export class AppError extends Error {
  constructor(public message: string, public statusCode = 400) {
    super(message);
    this.name = 'AppError';
  }
}

export function errorHandler(
  err: unknown,
  req: Request,
  res: Response,
  next: NextFunction
) {
  console.error('[Error]', err);

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
    });
  }

  if (err instanceof AppError) {
    return res.status(err.statusCode).json({
      success: false,
      code: err.statusCode,
      message: err.message,
    });
  }

  // Fallback for unknown errors
  res.status(500).json({
    success: false,
    code: 500,
    message: 'Something went wrong',
    ...(process.env.NODE_ENV === 'development' && { error: err }),
  });
}
