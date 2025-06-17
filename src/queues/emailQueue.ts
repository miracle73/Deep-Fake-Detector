import { Queue, QueueEvents } from 'bullmq';

import { redisConnection } from '../config/redis.js';
import logger from '../utils/logger.js';

const emailQueue = new Queue('emailQueue', {
  connection: redisConnection,
});

export const emailQueueEvent = new QueueEvents('emailQueueEvent', {
  connection: redisConnection,
});

emailQueueEvent.on('failed', ({ jobId, failedReason }) => {
  logger.error(`Job ${jobId} failed with error ${failedReason}`);
});

emailQueueEvent.on('waiting', ({ jobId }) => {
  logger.info(`A job with ID ${jobId} is waiting`);
});

emailQueueEvent.on('completed', ({ jobId }) => {
  logger.info(`Job ${jobId} completed`);
});

export default emailQueue;
