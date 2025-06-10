import { NextFunction, Response } from 'express';
import { AuthRequest } from './auth';
import Subscription from 'models/Subscription';

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
