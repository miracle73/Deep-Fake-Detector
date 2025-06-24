import type { NextFunction, Request, Response } from 'express';
import { AppError, AuthenticationError } from './error.js';
import logger from './logger.js';
import User from '../models/User.js';
import mongoose from 'mongoose';

export const validateAndDecrementQuota = async (
  req: Request,
  res: Response,
  next: NextFunction
) => {
  const session = await mongoose.startSession();
  session.startTransaction();

  try {
    const userId = req.user?._id;

    if (!userId || !mongoose.Types.ObjectId.isValid(userId)) {
      throw new AuthenticationError('Invalid user identifier');
    }

    if (req.user?.plan === 'SafeGuard Max' && req.user?.unlimitedQuota) {
      return next();
    }

    const updatedUser = await User.findOneAndUpdate(
      {
        _id: userId,
        'usageQuota.remainingAnalysis': { $gte: 1 },
      },
      {
        $inc: { 'usageQuota.remainingAnalysis': -1 },
        $set: { 'usageQuota.lastUsedAt': new Date() },
      },
      {
        new: true,
        session,
        select: 'usageQuota plan stripeSubscriptionId',
      }
    );

    if (!updatedUser) {
      const currentUser = await User.findById(userId)
        .select('usageQuota plan')
        .session(session);

      if (!currentUser) {
        throw new AuthenticationError('User not found');
      }

      throw new AppError(
        429,
        currentUser.plan === 'SafeGuard Free'
          ? 'Free tier quota exhausted. Upgrade to PRO for more analyses.'
          : 'Monthly quota exceeded. Reset occurs on 1st of each month.',
        {
          code: 'QUOTA_EXHAUSTED',
          resetDate: getMonthlyResetDate(),
          upgradeUrl: '/billing/upgrade',
          currentQuota: currentUser.usageQuota,
        }
      );
    }

    req.user = updatedUser;

    await session.commitTransaction();
    next();
  } catch (error) {
    await session.abortTransaction();

    logger.error('Quota validation error', {
      error,
      userId: req.user?._id,
      endpoint: req.originalUrl,
    });

    next(error);
  } finally {
    session.endSession();
  }
};

function getMonthlyResetDate(): Date {
  const now = new Date();
  return new Date(now.getFullYear(), now.getMonth() + 1, 1);
}
