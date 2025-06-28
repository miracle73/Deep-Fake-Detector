import { config } from 'dotenv';
import mongoose from 'mongoose';

import { resetMonthlyQuotas } from '../services/quotaReset.service.js';
import logger from '../utils/logger.js';

config();

async function test() {
  const DB_URI = process.env.MONGO_URI;

  if (!DB_URI) {
    logger.error('Provide DB connection string');
    process.exit(0);
  }

  await mongoose.connect(DB_URI);

  if (!mongoose.connection.db) {
    throw new Error('Database connection is not established.');
  }
  await mongoose.connection.db
    .collection('users')
    .updateMany(
      { plan: 'SafeGuard_Free' },
      { $set: { 'usageQuota.remainingAnalysis': 0 } }
    );

  await resetMonthlyQuotas();

  const users = await mongoose.connection.db
    .collection('users')
    .find({ plan: 'SafeGuard_Free' })
    .toArray();

  console.log(
    'Reset results:',
    users.map((u) => ({
      email: u.email,
      remaining: u.usageQuota.remainingAnalysis,
      lastReset: u.usageQuota.lastResetAt,
    }))
  );

  process.exit(0);
}

test();
