import * as WaitlistService from '../services/waitlist.js';
import { ConflictError, NotFoundError } from '../utils/error.js';
import logger from '../utils/logger.js';

import type { NextFunction, Request, Response } from 'express';
import type {
  WaitlistSignup,
  WaitlistStatus,
} from '../lib/schemas/waitlist.schema';

export const SignupForWaitlist = async (
  req: Request,
  res: Response,
  next: NextFunction
) => {
  try {
    const { email } = req.body as WaitlistSignup;

    const ipAddress = req.ip || req.socket.remoteAddress || 'unknown';
    const userAgent = req.get('User-Agent');

    logger.info('Waitlist signup attempt', { email, ipAddress, userAgent });

    const result = await WaitlistService.addToWaitlist({
      email,
      ipAddress,
      userAgent,
    });

    logger.info('Waitlist signup successful', {
      email,
      position: result.position,
      totalSignups: result.totalSignups,
    });

    res.status(201).json({
      success: true,
      message: 'Successfully joined the waitlist!',
      data: {
        email: result.email,
        position: result.position,
        estimatedWaitTime: result.estimatedWaitTime,
        totalSignups: result.totalSignups,
        signupDate: result.signupDate,
      },
    });
  } catch (error) {
    if (error instanceof ConflictError) {
      logger.warn('Duplicate waitlist signup attempt', {
        email: req.body.email,
        ip: req.ip,
      });

      res.status(409).json({
        success: false,
        error: 'DUPLICATE_EMAIL',
        message: 'This email is already on the waitlist.',
      });
    }

    logger.error('Waitlist signup error', {
      error: error instanceof Error ? error.message : 'Unknown error occured',
      email: req.body.email,
    });
    next(error);
  }
};

export async function getWaitlistStatus(
  req: Request,
  res: Response,
  next: NextFunction
) {
  try {
    const { email } = req.query as WaitlistStatus;

    logger.info('Waitlist status check', { email });

    const status = await WaitlistService.getWaitlistStatus(email);

    if (!status) {
      logger.info('Email not found in waitlist', { email });
      throw new NotFoundError('Email not found in waitlist');
    }

    logger.info('Waitlist status retrieved', {
      email,
      position: status.currentPosition,
    });

    res.json({
      success: true,
      data: {
        email: status.email,
        currentPosition: status.currentPosition,
        originalPosition: status.originalPosition,
        signupDate: status.signupDate,
        status: status.status,
        estimatedWaitTime: status.estimatedWaitTime,
        totalActive: status.totalActive,
      },
    });
  } catch (error) {
    logger.error('Waitlist status error', {
      error: error instanceof Error ? error.message : 'Unknown error occured',
      email: req.query.email,
    });
    next(error);
  }
}

export async function getWaitlistStats(
  req: Request,
  res: Response,
  next: NextFunction
) {
  try {
    const stats = await WaitlistService.getWaitlistStats();

    res.json({
      success: true,
      data: stats,
    });
  } catch (error) {
    logger.error('Waitlist stats error', {
      error: error instanceof Error ? error.message : 'Unknown error occured',
    });
    next(error);
  }
}
