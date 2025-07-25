import { Worker } from 'bullmq';

import { redisConnection } from '../config/redis.js';
import { Notification } from '../models/Notification.js';
import User from '../models/User.js';
import logger from '../utils/logger.js';

import type { NotificationJob } from '../types/queues.d.js';

logger.info('🚀 Notification worker is running...');

const mainWorkerOptions = {
  connection: redisConnection,
  concurrency: 5,
  removeOnFail: { count: 0 },
};

export const notificationWorker = new Worker(
  'notificationQueue',
  async (job) => {
    logger.info(job.name);
    logger.info(job.data);
    pushNotification(job);
  },
  mainWorkerOptions
);

notificationWorker.on('error', (err) => {
  logger.error(`Error processing job: ${err}`);
});

notificationWorker.on('completed', (job) => {
  logger.info(`✅ Notification job ${job.name} completed`);
});

notificationWorker.on('failed', (job, err) => {
  logger.error(`❌ Notification job ${job?.name} failed:`, err);
});

const pushNotification = async (job: NotificationJob) => {
  const { title, body, userId } = job.data;

  const user = await User.findById(userId);

  if (!user) {
    logger.error(`User with ID ${userId} not found for notification`);
    return;
  }

  const notification = await Notification.create(job.data);

  user.notifications.push(notification._id.toString());
  await user.save();

  return;
};
