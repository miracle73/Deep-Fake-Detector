import type { NextFunction, Response } from 'express';

import type { AuthRequest } from './auth.js';

const checkMediaConsent = async (
  req: AuthRequest,
  res: Response,
  next: NextFunction
) => {
  try {
    const user = req.user;

    if (!user) {
      return res.status(401).json({ error: 'Authentication required' });
    }

    if (!req.path.includes('/upload') && !req.path.includes('/process')) {
      return next();
    }

    if (!user.consent?.storeMedia) {
      return res.status(403).json({
        error: 'Media storage consent required',
        code: 'CONSENT_REQUIRED',
        solution: 'Update your consent at PATCH /api/user/consent/media',
      });
    }

    next();
  } catch (error) {
    console.error('Consent check failed:', error);
    res.status(500).json({ error: 'Consent verification failed' });
  }
};
