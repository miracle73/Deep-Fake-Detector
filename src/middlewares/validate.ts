import type { Request, Response, NextFunction } from 'express';
import { ValidationError } from 'utils/error';
import type { AnyZodObject, ZodSchema, ZodTypeAny } from 'zod';

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

export const validate = (schema: AnyZodObject) => {
  return async (req: Request, res: Response, next: NextFunction) => {
    try {
      await schema.parseAsync({
        body: req.body,
        query: req.query,
        params: req.params,
      });
      next();
    } catch (error) {
      next(
        new ValidationError(
          error instanceof Error ? error.message : 'Unknown error'
        )
      );
    }
  };
};
