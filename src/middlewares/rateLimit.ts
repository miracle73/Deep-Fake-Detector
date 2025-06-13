import { rateLimit } from 'express-rate-limit';
import type { Request, Response } from 'express';

export const limiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 100,
});

export const waitlistSignupLimiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 3,
  message: {
    success: false,
    error: 'RATE_LIMIT_EXCEEDED',
    message: 'Too many signup attempts. Please try again in 15 minutes.',
  },
  standardHeaders: true,
  legacyHeaders: false,
  keyGenerator: (req: Request) => {
    const email = req.body?.email || '';
    return `${req.ip}-${email}`;
  },
  skip: (req: Request) => {
    return false;
  },
});

export const waitlistStatusLimiter = rateLimit({
  windowMs: 5 * 60 * 1000,
  max: 20,
  message: {
    success: false,
    error: 'RATE_LIMIT_EXCEEDED',
    message:
      'Too many status check requests. Please try again in a few minutes.',
  },
  standardHeaders: true,
  legacyHeaders: false,
});

export const generalApiLimiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 100,
  message: {
    success: false,
    error: 'RATE_LIMIT_EXCEEDED',
    message: 'Too many requests. Please try again later.',
  },
  standardHeaders: true,
  legacyHeaders: false,
});
