import mongoose from 'mongoose';
import cron from 'node-cron';

import User from '../models/User.js';
import { getMonthlyResetDate } from '../utils/dateUtils.js';
import logger from '../utils/logger.js';

interface QuotaResetConfig {
  plan: string;
  monthlyAllocation: number;
  carryOver?: boolean;
}

const QUOTA_CONFIG: Record<string, QuotaResetConfig> = {
  free: {
    plan: 'SafeGuard_Free',
    monthlyAllocation: 3,
  },
  pro: {
    plan: 'SafeGuard_Pro',
    monthlyAllocation: 30,
    carryOver: true,
  },
  max: {
    plan: 'SafeGuard_Max',
    monthlyAllocation: 500,
    carryOver: true,
  },
};

export async function resetMonthlyQuotas() {
  const session = await mongoose.startSession();
  session.startTransaction();

  try {
    const resetDate = getMonthlyResetDate();

    const bulkOps = Object.entries(QUOTA_CONFIG).map(([plan, config]) => ({
      updateMany: {
        filter: {
          plan,
          unlimitedQuota: { $ne: true },
          $or: [
            { 'usageQuota.lastResetAt': { $lt: resetDate } },
            { 'usageQuota.lastResetAt': { $exists: false } },
          ],
        },
        update: {
          $set: {
            'usageQuota.remainingAnalysis': config.carryOver
              ? {
                  $min: [
                    config.monthlyAllocation,
                    {
                      $add: [
                        '$usageQuota.remainingAnalysis',
                        config.monthlyAllocation,
                      ],
                    },
                  ],
                }
              : config.monthlyAllocation,
            'usageQuota.lastResetAt': resetDate,
            'usageQuota.monthlyAnalysis': config.monthlyAllocation,
          },
        },
        session,
      },
    }));

    await User.bulkWrite(bulkOps, { session });

    await User.updateMany(
      {
        userType: 'enterprise',
        unlimitedQuota: false,
        'company.customQuota': { $exists: true },
      },
      {
        $set: {
          'usageQuota.remainingAnalysis': '$company.customQuota',
          'usageQuota.lastResetAt': resetDate,
        },
      },
      { session }
    );

    await session.commitTransaction();
    logger.info(`Successfully reset quotas for ${resetDate.toISOString()}`);
  } catch (error) {
    await session.abortTransaction();
    logger.error('Quota reset failed', error);
    throw error;
  } finally {
    session.endSession();
  }
}

export function startQuotaResetSchedule() {
  cron.schedule(
    '0 0 1 * *',
    async () => {
      logger.info('Starting monthly quota reset...');
      try {
        await resetMonthlyQuotas();
      } catch (error) {
        logger.error('Cron job failed', error);
      }
    },
    {
      timezone: 'UTC',
      name: 'monthly_quota_reset',
    }
  );
}
