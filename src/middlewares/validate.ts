import type { Request, Response, NextFunction } from 'express';
import { ValidationError } from '../utils/error.js';
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

    // req.validatedBody = result.data;
    req.body = result.data;

    next();
  };
}

export function validateQuery<T extends ZodTypeAny>(schema: T) {
  return (req: Request, res: Response, next: NextFunction) => {
    const result = schema.safeParse(req.query);

    console.log('this is result', result);

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
      return;
    }

    // req.query = result.data;
    // biome-ignore lint/suspicious/noExplicitAny: <explanation>
    (req as any).validatedQuery = result.data;

    next();
  };
}

export function validateParams<T extends ZodTypeAny>(schema: T) {
  return (req: Request, res: Response, next: NextFunction) => {
    const result = schema.safeParse(req.params);

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

    req.params = result.data;

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
