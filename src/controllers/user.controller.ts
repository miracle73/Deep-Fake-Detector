import { compare } from 'bcryptjs';

import User from '../models/User.js';
import { NotFoundError, ValidationError } from '../utils/error.js';
import logger from '../utils/logger.js';

import type { NextFunction, Request, Response } from 'express';
import type { AuthRequest } from '../middlewares/auth.js';
import type {
  VerifyMediaConsentInput,
  VerifySubUpdateInput,
} from '../lib/schemas/user.schema.js';
import Analysis from '../models/Analysis.js';

export async function getCurrentUser(
  req: AuthRequest,
  res: Response,
  next: NextFunction
) {
  try {
    const userId = req.user._id;

    const user = await User.findById(userId);

    if (!user) {
      throw new NotFoundError('User not found');
    }

    res.status(200).json({
      success: true,
      message: 'User fetched successfully',
      data: {
        user,
      },
    });
  } catch (error) {
    logger.error('Failed to fetch user', error);
    next(error);
  }
}

export async function UpdateUser(
  req: AuthRequest,
  res: Response,
  next: NextFunction
) {
  try {
    const userId = req.user?._id;

    const updateData = req.body;

    const user = await User.findById(userId);
    if (!user) {
      throw new NotFoundError('User not found');
    }

    if (updateData.email && updateData.email !== user.email) {
      const existingUser = await User.findOne({ email: updateData.email });
      if (existingUser) {
        throw new ValidationError('Email already in use');
      }
    }

    if (updateData.currentPassword && updateData.newPassword) {
      const isValidPassword = await compare(
        updateData.currentPassword,
        user.password
      );
      if (!isValidPassword) {
        throw new ValidationError('Current password is incorrect');
      }
      user.password = updateData.newPassword;
      delete updateData.currentPassword;
      delete updateData.newPassword;
    }

    Object.assign(user, updateData);
    await user.save();

    const updatedUser = await User.findById(userId).select('-password');

    res.status(200).json({
      success: true,
      message: 'User updated successfully',
      data: {
        updatedUser,
      },
    });
  } catch (error) {
    logger.error('Failed to update user', error);
    next(error);
  }
}

export async function DeleteUser(
  req: AuthRequest,
  res: Response,
  next: NextFunction
) {
  try {
    const userId = req.user?._id;

    const deletedUser = await User.findByIdAndDelete(userId);

    if (!deletedUser) {
      throw new NotFoundError('User not found');
    }

    res.status(200).json({
      success: true,
      message: 'User deleted successfully',
      data: {
        deletedUser,
      },
    });
  } catch (error) {
    logger.error('Failed to delete user', error);
    next(error);
  }
}

export const GetBillingHistory = async (
  req: Request,
  res: Response,
  next: NextFunction
) => {
  try {
    const result = req.user?.billingHistory;

    res.status(200).json({
      success: true,
      message: 'Billing history fetched successfully',
      data: result,
    });
  } catch (error) {
    logger.error('Error fetching billing history', error);
    next(error);
  }
};

export const getAnalysisHistory = async (
  req: AuthRequest,
  res: Response,
  next: NextFunction
) => {
  try {
    const userId = req.user.id;

    const analyses = await Analysis.find({ userId }).lean();

    res.status(200).json({
      success: true,
      analysisHistory: analyses,
    });
  } catch (error) {
    console.error('Failed to fetch analysis history:', error);
    next(error);
  }
};

export const updateSubscription = async (
  req: AuthRequest,
  res: Response,
  next: NextFunction
) => {
  try {
    const { emailSubscribed } = req.body as VerifySubUpdateInput;

    const userId = req.user.id;

    const user = await User.findByIdAndUpdate(
      userId,
      { emailSubscribed },
      { new: true }
    );

    res.status(200).json({
      success: true,
      message: 'Subscription updated successfully',
      data: {
        newStatus: user?.emailSubscribed,
      },
    });
  } catch (error) {
    console.error('Failed to update user subscription', error);
    next(error);
  }
};

export const updateMediaConsent = async (
  req: AuthRequest,
  res: Response,
  next: NextFunction
) => {
  try {
    const { allowStorage } = req.body as VerifyMediaConsentInput;

    const user = await User.findById(req.user.id);

    if (!user) {
      throw new NotFoundError('User not found');
    }

    user.consent = {
      storeMedia: allowStorage,
      updatedAt: new Date(),
    };

    await user.save();

    res.status(200).json({
      success: true,
      message: 'Media consent updated successfully',
      data: {
        consent: user?.consent,
      },
    });
  } catch (error) {
    console.error('Failed to update media consent:', error);
    next(error);
  }
};
