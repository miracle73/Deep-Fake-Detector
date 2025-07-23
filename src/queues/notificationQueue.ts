import { Queue, QueueEvents } from 'bullmq';

import { redisConnection } from '../config/redis.js';
import logger from '../utils/logger.js';

const notificationQueue = new Queue('notificationQueue', {
  connection: redisConnection,
});

export const notificationQueueEvent = new QueueEvents(
  'notificationQueueEvent',
  { connection: redisConnection }
);

notificationQueueEvent.on('failed', ({ jobId, failedReason }) => {
  logger.error(`Job ${jobId} failed with error ${failedReason}`);
});

notificationQueueEvent.on('waiting', ({ jobId }) => {
  logger.info(`A job with ID ${jobId} is waiting`);
});

notificationQueueEvent.on('completed', ({ jobId }) => {
  logger.info(`Job ${jobId} completed`);
});

export default notificationQueue;
