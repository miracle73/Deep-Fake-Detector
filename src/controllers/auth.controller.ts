import bcrypt from 'bcryptjs';
import crypto from 'node:crypto';

import { formatUserResponse } from '../lib/helpers.js';
import { Notification } from '../models/Notification.js';
import User from '../models/User.js';
import emailQueue from '../queues/emailQueue.js';
import { stripe } from '../services/stripeService.js';
import { sendPasswordResetEmail } from '../utils/email.js';
import {
  generateVerificationEmail,
  generateWelcomeEmail,
} from '../utils/email.templates.js';
import { AppError } from '../utils/error.js';
import { generateToken } from '../utils/generateToken.js';
import logger from '../utils/logger.js';
import {
  generateEmailVerificationToken,
  verifyEmailVerificationToken,
} from '../utils/token.js';

import type { z } from 'zod';

import type {
  individualUserSchema,
  enterpriseUserSchema,
} from '../lib/schemas/user.schema.js';

import type { NextFunction, Request, Response } from 'express';
import type {
  ForgotPasswordInput,
  LoginInput,
  RegisterInput,
} from '../lib/schemas/user.schema.js';
import type { AuthResponse, GoogleTempUser } from '../types/user.d.js';

type UserData = {
  email: string;
  password: string;
  userType: 'individual' | 'enterprise';
  agreedToTerms: boolean;
  termsAgreedAt: Date;
  plan: 'SafeGuard_Free' | 'SafeGuard_Pro' | 'SafeGuard_Max';
  isEmailVerified: boolean;
  stripeCustomerId?: string;
} & (
  | {
      userType: 'individual';
      firstName: string;
      lastName: string;
    }
  | {
      userType: 'enterprise';
      company: {
        name: string;
        website: string;
        size?: string;
        industry?: string;
      };
      billingContact: {
        name: string;
        email: string;
        phone: string;
      };
    }
);

export const register = async (
  req: Request,
  res: Response,
  next: NextFunction
) => {
  try {
    const {
      email,
      password,
      agreedToTerms,
      userType,
      plan = 'SafeGuard_Free',
    } = req.body as RegisterInput;

    if (!agreedToTerms) {
      throw new AppError(
        400,
        'You must agree to the terms and conditions',
        null
      );
    }

    const existingUser = await User.findOne({ email });
    if (existingUser) {
      throw new AppError(400, 'User with this email already exists', null);
    }

    let userData: UserData = {
      email,
      password,
      userType,
      agreedToTerms,
      termsAgreedAt: new Date(),
      plan,
      isEmailVerified: false,
    } as UserData;

    if (userType === 'individual') {
      const { firstName, lastName } = req.body as z.infer<
        typeof individualUserSchema
      >;
      userData = {
        ...userData,
        userType: 'individual',
        firstName,
        lastName,
      } as UserData & { userType: 'individual' };
    } else {
      const { company, billingContact } = req.body as z.infer<
        typeof enterpriseUserSchema
      >;
      userData = {
        ...userData,
        userType: 'enterprise',
        company,
        billingContact,
      } as UserData & { userType: 'enterprise' };
    }

    const stripeCustomer = await stripe.customers.create({
      email,
      name:
        userType === 'individual'
          ? `${(userData as UserData & { userType: 'individual' }).firstName} ${
              (userData as UserData & { userType: 'individual' }).lastName
            }`
          : (userData as UserData & { userType: 'enterprise' }).company.name,
      metadata: {
        appUserType: userType,
      },
    });

    if (!stripeCustomer) {
      throw new AppError(400, 'Failed to setup stripe for user', null);
    }

    userData.stripeCustomerId = stripeCustomer.id;
    const user = await User.create(userData);

    const createdNotification = await Notification.create({
      userId: user._id,
      type: 'system',
      title: 'Welcome to SafeGuard Media',
      message: 'Lorem ipsum dolor elistir',
      expiresAt: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000),
    });

    user.notifications.push(createdNotification._id.toString());
    await user.save();

    const token = generateToken(user._id.toString());

    const emailToken = generateEmailVerificationToken(user._id.toString());
    const verificationUrl = `${process.env.FRONTEND_URL}/verify-email?token=${emailToken}`;

    const verificationEmail = generateVerificationEmail({
      name: req.body.firstName || req.body.company.name,
      verificationUrl,
    });

    await emailQueue.add('verification-email', {
      to: email,
      subject: 'Welcome to SafeGuard Media – Verify Your Email',
      html: verificationEmail,
    });

    const response: AuthResponse = {
      success: true,
      token,
      user: formatUserResponse(user),
    };

    res.status(201).json(response);
  } catch (error) {
    logger.error('Failed to register user:', error);
    next(error);
  }
};

export const login = async (
  req: Request,
  res: Response,
  next: NextFunction
) => {
  try {
    const { email, password } = req.body as LoginInput;

    const user = await User.findOne({ email }).select('+password');
    if (!user) {
      return res.status(404).json({
        success: false,
        code: 404,
        message: 'User not found',
        details: null,
      });
    }

    const isPasswordMatch = await bcrypt.compare(password, user.password);
    if (!isPasswordMatch) {
      return res.status(401).json({
        success: false,
        code: 401,
        message: 'Invalid credentials',
        details: null,
      });
    }

    user.lastLogin = new Date();

    await user.save();

    const token = generateToken(user._id.toString());

    const response: AuthResponse = {
      success: true,
      token,
      user: formatUserResponse(user),
    };

    res.status(200).json(response);
  } catch (error) {
    logger.error('Failed to login user:', error);
    next(error);
  }
};

export const googleLogin = async (
  req: Request,
  res: Response,
  next: NextFunction
) => {
  try {
    const { email, googleId, firstName, lastName } = req.user as GoogleTempUser;
    const { userType, agreedToTerms, plan = 'free' } = req.body;

    if (!agreedToTerms) {
      throw new AppError(
        400,
        'You must agree to the terms and conditions',
        null
      );
    }

    let user = await User.findOne({ email });

    if (user) {
      if (!user.googleId) {
        user.googleId = googleId;
        user.isGoogleUser = true;

        if (user.userType === 'individual') {
          if (!user.firstName) user.firstName = firstName || 'Unknown';
          if (!user.lastName) user.lastName = lastName || 'Unknown';
        }

        await user.save();
      } else if (user.googleId !== googleId) {
        return res.status(400).json({
          success: false,
          message: 'Email already associated with different Google account',
        });
      }
    } else {
      const stripeCustomer = await stripe.customers.create({
        email,
        name:
          userType === 'individual'
            ? `${firstName} ${lastName}`
            : req.body.company?.name || email,
        metadata: {
          appUserType: userType,
        },
      });

      if (!stripeCustomer) {
        throw new AppError(400, 'Failed to setup stripe for user', null);
      }

      const userData = {
        email,
        userType,
        googleId,
        agreedToTerms: true,
        termsAgreedAt: new Date(),
        plan,
        stripeCustomerId: stripeCustomer.id,
        isEmailVerified: true,
        isGoogleUser: true,
        ...(userType === 'individual'
          ? { firstName, lastName }
          : {
              company: req.body.company,
              billingContact: req.body.billingContact,
            }),
      };

      user = await User.create(userData);
    }

    const token = generateToken(user._id.toString());

    const response: AuthResponse = {
      success: true,
      token,
      user: formatUserResponse(user),
    };

    res.status(200).json(response);
  } catch (error) {
    logger.error('Google sign-in failed', error);
    next(error);
  }
};

export const forgotPassword = async (
  req: Request,
  res: Response,
  next: NextFunction
) => {
  try {
    const { email } = req.body as ForgotPasswordInput;

    const user = await User.findOne({ email });

    if (!user) {
      return res.status(404).json({
        success: false,
        code: 404,
        message: 'No user found with that email',
        details: null,
      });
    }

    const resetToken = crypto.randomBytes(20).toString('hex');

    user.resetPasswordToken = crypto
      .createHash('sha256')
      .update(resetToken)
      .digest('hex');

    user.resetPasswordExpire = new Date(Date.now() + 10 * (60 * 1000));

    await user.save();

    const resetUrl = `${process.env.FRONTEND_URL}/reset-password/${resetToken}`;

    // const resetUrl = `${req.protocol}://${req.get(
    //   'host'
    // )}/api/v1/auth/reset-password/${resetToken}`;

    logger.info(resetUrl);

    await sendPasswordResetEmail(user.email, resetUrl);

    res.status(200).json({
      success: true,
      code: 200,
      message: 'Password reset email sent',
    });
  } catch (error) {
    logger.error('Failed to send password reset email:', error);
    next(error);
  }
};

export const resetPassword = async (
  req: Request,
  res: Response,
  next: NextFunction
) => {
  try {
    const { password } = req.body;

    const resetPasswordToken = crypto
      .createHash('sha256')
      .update(req.params.resetToken)
      .digest('hex');

    const user = await User.findOne({
      resetPasswordToken,
      resetPasswordExpire: { $gt: Date.now() },
    });

    if (!user) {
      return res.status(404).json({
        success: false,
        code: 404,
        message: 'Invalid or expired password reset token',
        details: null,
      });
    }

    const salt = await bcrypt.genSalt(10);
    user.password = await bcrypt.hash(password, salt);
    user.resetPasswordExpire = undefined;
    user.resetPasswordToken = undefined;
    user.passwordChangedAt = new Date();
    user.lastLogin = new Date();

    await user.save();

    const token = generateToken(user._id.toString());

    res.status(200).json({
      success: true,
      code: 200,
      message: 'Password reset successful',
      token,
      user: {
        id: user._id,
        email: user.email,
        userType: user.userType,
        // firstName: user.firstName,
        // lastName: user.lastName,
        plan: user.plan,
      },
    });
  } catch (error) {
    logger.error('Failed to reset password:', error);
    next(error);
  }
};

export const verifyEmail = async (
  req: Request,
  res: Response,
  next: NextFunction
) => {
  try {
    const { token } = req.validatedQuery;

    if (!token || typeof token !== 'string') {
      throw new AppError(400, 'Invalid or missing verification token', null);
    }

    const { userId } = verifyEmailVerificationToken(token);

    if (!userId) {
      throw new AppError(400, 'Invalid or expired verification token', null);
    }

    const user = await User.findById(userId);

    console.log(user);
    if (!user) throw new AppError(404, 'User not found', null);

    if (user.isEmailVerified) {
      return res
        .status(200)
        .json({ success: true, message: 'Email already verified.' });
    }

    user.isEmailVerified = true;
    await user.save();

    let displayName = 'there';
    if (
      'firstName' in user &&
      typeof user.firstName === 'string' &&
      user.firstName
    ) {
      displayName = user.firstName;
    } else if (
      'company' in user &&
      user.company &&
      typeof user.company === 'object' &&
      'name' in user.company &&
      typeof user.company.name === 'string'
    ) {
      displayName = user.company.name;
    }

    const welcomeEmail = generateWelcomeEmail({
      name: displayName,
    });

    await emailQueue.add('verification-email', {
      to: user.email,
      subject: 'Welcome to SafeGuard Media',
      html: welcomeEmail,
    });

    res
      .status(200)
      .json({ success: true, message: 'Email verified successfully!' });
  } catch (error) {
    logger.error('Email verification failed:', error);
    next(error);
  }
};

export const resendVerificationEmail = async (
  req: Request,
  res: Response,
  next: NextFunction
) => {
  try {
    const { email } = req.body;

    if (!email) {
      throw new AppError(400, 'Email is required', null);
    }

    const user = await User.findOne({ email });

    if (!user) {
      throw new AppError(404, 'No user found with that email', null);
    }

    if (user.isEmailVerified) {
      throw new AppError(404, 'Email is already verified', null);
    }

    const token = generateToken(user._id.toString());
    const verificationUrl = `${process.env.FRONTEND_URL}/verify-email?token=${token}`;
    const html = generateVerificationEmail({
      name: 'there',
      verificationUrl,
    });

    await emailQueue.add('verification-email', {
      to: user.email,
      subject: 'Verify Your Email',
      html,
    });

    logger.info(`Resent verification email to: ${user.email}`);

    res.status(200).json({
      success: true,
      message: 'Verification email sent.',
    });
  } catch (error) {
    logger.error('Error in resendVerificationEmail:', error);
    next(error);
  }
};
