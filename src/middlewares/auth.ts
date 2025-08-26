import axios from 'axios';

import User from '../models/User.js';
import { validateSession, verifyToken } from '../services/auth.service.js';
import { AppError, AuthenticationError } from '../utils/error.js';
import logger from '../utils/logger.js';

import type { Request, Response, NextFunction } from 'express';
import type { UserRole } from '../types/roles.js';

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
    throw new AppError(401, 'Invalid Token', null);
  }

  try {
    const decoded = verifyToken(token);

    const isValid = await validateSession(
      decoded.userId,
      decoded.sessionVersion
    );

    if (!isValid) {
      res.status(401).json({ error: 'Session expired. Please login again.' });
      return;
    }
    // if (user.passwordChangedAt) {
    //   const changedTimestamp = user.passwordChangedAt.getTime() / 1000;
    // if (decoded.iat && changedTimestamp > decoded.iat) {
    //   throw new AppError(
    //     401,
    //     'User recently changed password. Please log in again'
    //   );
    // }
    // }

    const user = await User.findById(decoded.userId);

    if (!user) {
      throw new AppError(401, 'Not authorized to access this route', null);
    }

    req.user = user;
    next();
  } catch (error) {
    logger.error('Error verifying token:', error);
    next(error);
  }
};

export const authorize =
  (...roles: string[]) =>
  (req: Request, res: Response, next: NextFunction) => {
    if (!req.user || !roles.includes(req.user.userType)) {
      res.status(403).json({
        success: false,
        code: 403,
        message: 'You do not have permission to perform this action',
        details: null,
      });
    }
    next();
  };

export function authorizeRoles(...roles: string[]) {
  return (req: AuthRequest, res: Response, next: NextFunction) => {
    if (!req.user) {
      return next(new AuthenticationError('Not authenticated'));
    }

    if (!roles.includes(req.user.role)) {
      return next(new AuthenticationError('Not authorized'));
    }

    next();
  };
}

export const enterpriseOnly = (
  req: Request,
  res: Response,
  next: NextFunction
) => {
  if (!req.user || req.user.userType !== 'enterprise') {
    throw new AppError(
      403,
      'This feature is only available for teams and enterprise accounts',
      null
    );
  }

  next();
};

// export const requireRole = (...allowedRoles: UserRole[]) => {
//   return (req: Request, res: Response, next: NextFunction) => {
//     if (!req.user?.roles.some((role) => allowedRoles.includes(role))) {
//       return res.status(403).json({
//         success: false,
//         error: 'Forbidden',
//         message: `Requires one of these roles: ${allowedRoles.join(', ')}`,
//       });
//     }
//     next();
//   };
// };

export interface AuthRequest extends Request {
  // biome-ignore lint/suspicious/noExplicitAny: <explanation>
  user?: any;
  // biome-ignore lint/suspicious/noExplicitAny: <explanation>
  subscription?: any;
}

export const convertAccessTokenToIdToken = async (
  req: AuthRequest,
  res: Response,
  next: NextFunction
) => {
  try {
    const { access_token } = req.body.idToken; // This is actually the access token

    if (!access_token) {
      res.status(400).json({ message: 'Access Token required' });
    }

    // Fetch user info using the access token
    // const response = await axios.get(
    //   `https://www.googleapis.com/oauth2/v3/tokeninfo?access_token=${access_token}`
    // );

    const response = await axios.get(
      'https://www.googleapis.com/oauth2/v3/userinfo',
      { headers: { Authorization: `Bearer ${access_token}` } }
    );

    if (!response.data.email) {
      res.status(400).json({ message: 'Invalid Google Access Token' });
    }

    logger.info('response from google', response.data);

    req.user = {
      email: response.data.email,
      firstName: response.data.given_name,
      lastName: response.data.family_name,
      googleId: response.data.sub,
    };

    next();
  } catch (error) {
    const errorMessage =
      error instanceof Error ? error.message : 'Unknown error occured';
    logger.error('Token conversion error:', errorMessage);
    next(error);
  }
};
