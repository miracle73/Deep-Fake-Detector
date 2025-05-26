import type { Request, Response, NextFunction } from 'express';
import type { ZodSchema, ZodTypeAny } from 'zod';

export function validateInput<T extends ZodTypeAny>(schema: T) {
  return (req: Request, res: Response, next: NextFunction) => {
    const result = schema.safeParse(req.body);

    if (!result.success) {
      const errors = result.error.errors.map((e) => ({
        field: e.path.join('.') || 'unknown',
        message: e.message,
      }));

      res.status(400).json({
        success: false,
        code: 400,
        message: 'Validation errors',
        details: errors,
      });
    }

    req.body = result.data;

    next();
  };
}
