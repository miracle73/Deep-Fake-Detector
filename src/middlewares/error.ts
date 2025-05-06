import type { NextFunction, Request, Response } from 'express';
import ApiError from '../utils/error.js';

export const errorHandler = (
  err: {
    message: string;
    stack: any;
    code: number;
    success: boolean;
    details?: any;
  },
  req: Request,
  res: Response,
  next: NextFunction
) => {
  console.error('Error:', {
    message: err.message,
    // stack: err.stack,
    details: (err as any).details || 'No additional details',
  });

  // if (err instanceof ApiError) {
  //   res.status(err.code).json({
  //     success: err.success,
  //     status: 'error',
  //     code: err.code,
  //     message: err.message,
  //     details: err.details || 'No additional details',
  //   });
  //   return;
  // }

  res.status(500).json({
    success: err.success,
    status: 'error',
    code: err.code,
    message: err.message || 'Internal server error',
    details: err.details || 'No additional details',
  });
};
