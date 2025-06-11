import express from 'express';
import Subscription from 'models/Subscription';

import type { NextFunction, Response } from 'express';
import type { AuthRequest } from './auth';

export const checkSubscription = async (
  req: AuthRequest,
  res: Response,
  next: NextFunction
) => {
  const subscription = await Subscription.findOne({
    user: req.user._id,
    status: 'active',
  });
  if (!subscription) {
    return res
      .status(403)
      .json({ message: 'You must have an active subscription' });
  }
  req.subscription = subscription;
  next();
};

export const stripeWebhook = express.raw({ type: 'application/json' });
