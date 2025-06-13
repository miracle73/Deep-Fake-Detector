import bcrypt from 'bcryptjs';
import jwt from 'jsonwebtoken';
import crypto from 'node:crypto';
import { AppError } from 'utils/error.js';

import User from '../models/User.js';
import { stripe } from '../services/stripeService.js';
import { generateToken } from '../utils/generateToken.js';

import logger from '../utils/logger';

import type { NextFunction, Request, Response } from 'express';
import type { Secret } from 'jsonwebtoken';
import type {
  ForgotPasswordInput,
  LoginInput,
  RegisterInput,
} from '../lib/schemas/user.schema.js';

export const register = async (
  req: Request,
  res: Response,
  next: NextFunction
) => {
  try {
    const {
      email,
      password,
      firstName,
      lastName,
      agreedToTerms,
      userType,
      plan = 'free',
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

    const stripeCustomer = await stripe.customers.create({
      email,
      name: `${firstName} ${lastName}`,
      metadata: {
        appUserType: userType,
      },
    });

    if (!stripeCustomer) {
      throw new AppError(400, 'Failed to setup stripe for user', null);
    }

    const userData = {
      email,
      password,
      firstName,
      lastName,
      userType,
      agreedToTerms,
      termsAgreedAt: new Date(),
      plan,
      stripeCustomerId: stripeCustomer.id,
      isEmailVerified: false,
      ...(userType === 'enterprise'
        ? {
            company: req.body.company,
            billingContact: req.body.billingContact,
          }
        : {}),
    };

    const user = await User.create(userData);

    // const verificationToken = jwt.sign(
    //   { userId: user._id },
    //   process.env.JWT_SECRET as Secret,
    //   { expiresIn: '24h' }
    // );

    // const verificationUrl = `${process.env.FRONTEND_URL}/verify-email?token=${verificationToken}`;
    // console.log(verificationUrl);
    // await sendVerificationEmail(user.email, verificationToken);

    const token = generateToken(user._id.toString());

    res.status(201).json({
      success: true,
      token,
      user: {
        id: user._id,
        email: user.email,
        firstName: user.firstName,
        lastName: user.lastName,
        userType: user.userType,
        plan: user.plan,
      },
    });
  } catch (error) {
    logger.error('Failed to register user:', error);
    next(error);
  }
};

export const login = async (req: Request, res: Response) => {
  try {
    const { email, password } = req.body as LoginInput;

    const user = await User.findOne({ email }).select('+password');
    if (!user) {
      return res.status(401).json({
        success: false,
        code: 401,
        message: 'Invalid credentials',
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

    res.status(200).json({
      success: true,
      token,
      user: {
        id: user._id,
        email: user.email,
        firstName: user.firstName,
        lastName: user.lastName,
        userType: user.userType,
        plan: user.plan,
      },
    });
  } catch (error) {
    console.error('Failed to login user:', error);
    res.status(500).json({
      success: false,
      code: 500,
      message: 'Internal server error',
      details: null,
    });
  }
};

export const forgotPassword = async (req: Request, res: Response) => {
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

    const resetUrl = `${req.protocol}://${req.get(
      'host'
    )}/api/v1/auth/reset-password/${resetToken}`;

    logger.info(resetUrl);

    // send mail
    // await sendPasswordResetEmail(user.email, resetUrl);

    res.status(200).json({
      success: true,
      code: 200,
      message: 'Password reset email sent',
    });
  } catch (error) {
    console.error('Failed to send password reset email:', error);
    res.status(500).json({
      success: false,
      code: 500,
      message: 'Internal server error',
      details: null,
    });
  }
};

export const resetPassword = async (req: Request, res: Response) => {
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
        firstName: user.firstName,
        lastName: user.lastName,
        plan: user.plan,
      },
    });
  } catch (error) {
    console.error('Failed to reset password:', error);
    res.status(500).json({
      success: false,
      code: 500,
      message: 'Internal server error',
      details: null,
    });
  }
};
