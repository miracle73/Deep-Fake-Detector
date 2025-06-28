import User from '../models/User.js';

console.log('MIGRATING....');

async function migrate() {
  await User.updateMany(
    {
      isActive: true,
      currentPeriodEnd: { $exists: false },
    },
    {
      $set: {
        currentPeriodEnd: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000),
      },
    }
  );
}

console.log('Migration started...');
console.log(
  'This will set currentPeriodEnd to 30 days from now for active users without it.'
);

migrate().then(() => process.exit(0));
