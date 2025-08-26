import { Request, Response, NextFunction } from 'express';

export interface CustomSession {
  userId?: string;
  accessCode?: string;
  destroy(callback: (err: any) => void): void;
}

export const requireAuth = (
  req: Request,
  res: Response,
  next: NextFunction
) => {
  if (req.session && (req.session as CustomSession).userId) {
    return next();
  }
  res.status(401).json({ error: 'Authentication required' });
};

export const preventReLogin = (
  req: Request,
  res: Response,
  next: NextFunction
) => {
  if (req.session && (req.session as CustomSession).userId) {
    return res
      .status(403)
      .json({ error: 'Already authenticated. Please logout first.' });
  }
  next();
};
