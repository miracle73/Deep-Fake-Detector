import jwt from 'jsonwebtoken';
import User from '../models/User.js';
import type { Request, Response, NextFunction } from 'express';

export const protect = async (
  req: Request,
  res: Response,
  next: NextFunction
) => {
  let token: string | undefined;

  const authorization = req.headers.authorization;

  if (authorization?.startsWith('Bearer')) {
    token = authorization.split(' ')[1];
  }

  if (!token) {
    return res.status(401).json({
      success: false,
      code: 401,
      message: 'Not authorized to access this route',
      details: null,
    });
  }

  try {
    const decoded = (await jwt.verify(
      token,
      process.env.JWT_SECRET as string
    )) as { id: string; iat: number };

    const user = await User.findById(decoded.id);
    if (!user) {
      return res.status(401).json({
        success: false,
        code: 401,
        message: 'Not authorized to access this route',
        details: null,
      });
    }

    if (user.passwordChangedAt) {
      const changedTimestamp = user.passwordChangedAt.getTime() / 1000;
      if (decoded.iat && changedTimestamp > decoded.iat) {
        return res.status(401).json({
          success: false,
          code: 401,
          message: 'User recently changed password. Please log in again',
          details: null,
        });
      }
    }

    req.user = user;
    next();
  } catch (error) {
    console.error('Error verifying token:', error);
    return res.status(401).json({
      success: false,
      code: 401,
      message: 'Not authorized to access this route',
      details: null,
    });
  }
};

export const authorize =
  (...roles: string[]) =>
  (req: Request, res: Response, next: NextFunction) => {
    if (!req.user || !roles.includes(req.user.userType)) {
      return res.status(403).json({
        success: false,
        code: 403,
        message: 'You do not have permission to perform this action',
        details: null,
      });
    }
    next();
  };

export const enterpriseOnly = (
  req: Request,
  res: Response,
  next: NextFunction
) => {
  if (!req.user || req.user.userType !== 'enterprise') {
    return res.status(403).json({
      success: false,
      code: 403,
      message:
        'This feature is only available for teams and enterprise accounts',
      details: null,
    });
  }

  next();
};
