import {
  calculateEstimatedWaitTime,
  sendInvitationEmail,
  sendWelcomeEmail,
} from '../lib/helpers.js';
import Waitlist from '../models/Waitlist.js';
import { ConflictError } from '../utils/error.js';
import logger from '../utils/logger.js';

import type {
  WaitlistSignupData,
  WaitlistSignupResult,
  WaitlistStatusResult,
} from '../types/waitlist.d';

export async function addToWaitlist(
  data: WaitlistSignupData
): Promise<WaitlistSignupResult> {
  const { email, ipAddress, userAgent } = data;

  const existingEntry = await Waitlist.findOne({ email });
  if (existingEntry) {
    throw new ConflictError('Email already exists in waitlist');
  }

  const position = await (Waitlist as any).getNextPosition();

  const waitlistEntry = await Waitlist.create({
    email,
    position,
    ipAddress,
    userAgent,
    signupDate: new Date(),
  });

  const totalSignups = await Waitlist.countDocuments();

  try {
    await sendWelcomeEmail({
      email,
      position,
      estimatedWaitTime: calculateEstimatedWaitTime(position),
    });
  } catch (emailError) {
    logger.error('Failed to send welcome email', {
      email,
      error:
        emailError instanceof Error
          ? emailError.message
          : 'Unknown error occurred',
    });
  }

  return {
    email: waitlistEntry.email,
    position: waitlistEntry.position,
    signupDate: waitlistEntry.joinedAt,
    estimatedWaitTime: calculateEstimatedWaitTime(position),
    totalSignups,
  };
}

export async function getWaitlistStatus(
  email: string
): Promise<WaitlistStatusResult | null> {
  const entry = await Waitlist.findOne({ email });
  if (!entry) {
    return null;
  }

  const currentPosition = await (Waitlist as any).getUserPosition(email);
  const totalActive = await Waitlist.countDocuments({ status: 'active' });

  return {
    email: entry.email,
    currentPosition: currentPosition || entry.position,
    originalPosition: entry.position,
    signupDate: entry.joinedAt,
    status: entry.status,
    estimatedWaitTime: calculateEstimatedWaitTime(
      currentPosition || entry.position
    ),
    totalActive,
  };
}

export async function getWaitlistStats() {
  const [totalSignups, activeUsers, invitedUsers, convertedUsers] =
    await Promise.all([
      Waitlist.countDocuments(),
      Waitlist.countDocuments({ status: 'active' }),
      Waitlist.countDocuments({ status: 'invited' }),
      Waitlist.countDocuments({ status: 'converted' }),
    ]);

  const recentSignups = await Waitlist.countDocuments({
    signupDate: { $gte: new Date(Date.now() - 24 * 60 * 60 * 1000) },
  });

  const conversionRate =
    totalSignups > 0 ? (convertedUsers / totalSignups) * 100 : 0;

  return {
    totalSignups,
    activeUsers,
    invitedUsers,
    convertedUsers,
    recentSignups,
    conversionRate: Math.round(conversionRate * 100) / 100,
  };
}

export async function inviteNextUsers(count = 1) {
  const usersToInvite = await Waitlist.find({ status: 'active' })
    .sort({ position: 1 })
    .limit(count);

  const invitedUsers = [];

  for (const user of usersToInvite) {
    user.status = 'invited';
    user.invitedAt = new Date();
    await user.save();

    try {
      await sendInvitationEmail(user.email);
      invitedUsers.push(user);
      logger.info('User invited from waitlist', { email: user.email });
    } catch (error) {
      logger.error('Failed to send invitation email', {
        email: user.email,
        error: error instanceof Error ? error.message : 'Unknown error occured',
      });
    }
  }

  return invitedUsers;
}
