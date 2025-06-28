import type { NextFunction, Request, Response } from 'express';
import User from '../models/User.js';
import { AppError } from '../utils/error.js';

export const checkActivePlan = async (
  req: Request,
  res: Response,
  next: NextFunction
) => {
  const userId = req.user?._id;

  if (!userId) {
    return next(new AppError(401, 'Authentication required'));
  }

  const user = await User.findById(userId).select(
    'plan isActive stripeSubscriptionId'
  );

  if (!user) {
    return next(new AppError(404, 'User not found'));
  }

  // allow if user has no subscription (first-time purchase)
  if (!user.stripeCustomerId) {
    return next();
  }

  if (user.isActive) {
    return next(
      new AppError(
        400,
        'You already have an active subscription. Manage your plan in billing settings.',
        {
          redirectUrl: '/account/billing',
          renewalDate: user.currentPeriodEnd,
        }
      )
    );
  }

  next();
};
