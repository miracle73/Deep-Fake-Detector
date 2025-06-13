import { compare } from 'bcryptjs';

import User from '../models/User.js';
import { NotFoundError, ValidationError } from '../utils/error.js';
import logger from '../utils/logger.js';

import type { NextFunction, Response } from 'express';
import type { AuthRequest } from '../middlewares/auth.js';

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
