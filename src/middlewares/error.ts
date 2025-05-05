import { NextFunction, Request, Response } from 'express';

class ApiError extends Error {
  constructor(
    public code: number,
    public success: boolean,
    public message: string,
    public details: any
  ) {
    super(message);
  }
}

export const errorHandler = (
  err: { message: any; stack: any; code: number; success: any; details: any },
  req: Request,
  res: Response,
  next: NextFunction
) => {
  console.error('Error:', {
    message: err.message,
    stack: err.stack,
    details: (err as any).details || 'No additional details',
  });

  if (err instanceof ApiError) {
    res.status(err.code).json({
      success: err.success,
      status: 'error',
      code: err.code,
      message: err.message,
      details: err.details || null,
    });
    return;
  }

  res.status(500).json({
    success: false,
    status: 'error',
    code: 500,
    message: 'Internal server error',
  });
};
