import { stripe } from '../services/stripeService.js';

import { formatUserResponse } from '../lib/helpers.js';
import DemoRequest from '../models/DemoRequest.js';
import User from '../models/User.js';
import notificationQueue from '../queues/notificationQueue.js';
import { createDemoUser } from '../services/demoRequest.service.js';
import { AppError, NotFoundError } from '../utils/error.js';
import logger from '../utils/logger.js';

import type { Request, NextFunction, Response } from 'express';
import type {
  DemoRequest as DemoRequestSchema,
  DemoUser,
} from '../lib/schemas/demo.schema.js';
import type { UserData } from './auth.controller.js';
import type { AuthResponse } from '../types/user.d.js';

export const Submit = async (
  req: Request,
  res: Response,
  next: NextFunction
) => {
  try {
    const user = await createDemoUser(req.body as DemoRequestSchema);

    // const html = generateDemoConfirmationEmail(user.firstName);

    // await emailQueue.add('demoConfirmationEmail', {
    //   to: user.email,
    //   subject: 'Demo User Created Successfully',
    //   html,
    // });

    res.status(201).json({
      success: true,
      message: 'Demo user created successfully',
      data: user,
    });
  } catch (error) {
    logger.error('Error in Submit controller:', error);
    next(error);
  }
};

export const completeProfile = async (
  req: Request,
  res: Response,
  next: NextFunction
) => {
  try {
    const { email, password } = req.body as DemoUser;

    const existingUser = await DemoRequest.findOne({ email });

    if (!existingUser) {
      throw new NotFoundError('User not found. Please sign up');
    }

    const userData: UserData = {
      email,
      password,
      userType: 'individual',
      agreedToTerms: true,
      termsAgreedAt: new Date(),
      plan: 'SafeGuard_Free',
      isEmailVerified: true,
      firstName: existingUser.firstName,
      lastName: existingUser.lastName,
    } as UserData;

    const stripeCustomer = await stripe.customers.create({
      email,
      name: `${existingUser.firstName} ${existingUser.lastName}`,
      metadata: {
        appuserType: 'individual',
      },
    });

    if (!stripeCustomer) {
      throw new AppError(400, 'Failed to setup stripe for user', null);
    }

    userData.stripeCustomerId = stripeCustomer.id;

    const user = await User.create(userData);

    if (!user) {
      throw new AppError(500, 'Failed to create user', null);
    }

    await notificationQueue.add('Welcome-Notification', {
      userId: user._id,
      type: 'system',
      title: 'Welcome to SafeGuard Media',
      message:
        'Go ahead and instantly analyze media for AI manipulation with our robust and easy-to-use platform',
      expiresAt: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000),
    });

    const response: AuthResponse = {
      success: true,
      user: formatUserResponse(user),
    };

    res.status(201).json(response);
  } catch (error) {
    logger.error('Failed to register user:', error);
    next(error);
  }
};
