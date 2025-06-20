// Use Cloud Scheduler (GCP) or Agenda.js
import User from '../models/User';
import cron from 'node-cron';

cron.schedule('0 0 1 * *', async () => {
  await User.updateMany(
    { 'usageQuota.monthlyAnalysis': { $gt: 0 } },
    {
      $set: {
        'usageQuota.remainingAnalysis': '$usageQuota.monthlyAnalysis',
        'usageQuota.lastResetAt': new Date(),
      },
    }
  );
});
