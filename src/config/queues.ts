import { emailWorker } from '../workers/emailWorker.js';
import emailQueue, { emailQueueEvent } from '../queues/emailQueue.js';
import logger from '../utils/logger.js';
import notificationQueue from 'queues/notificationQueue.js';
import { notificationWorker } from 'workers/notificationWorker.js';

export const startQueues = async () => {
  await emailQueue.waitUntilReady();
  await emailWorker.waitUntilReady();
  await emailQueueEvent.waitUntilReady();
  await notificationQueue.waitUntilReady();
  await notificationWorker.waitUntilReady();
  await emailQueueEvent.waitUntilReady();
  logger.info('Queues and workers are ready!');
};

export const stopQueues = async () => {
  await emailQueue.close();
  await emailWorker.close();
  await notificationQueue.close();
  await notificationWorker.close();
  logger.info('Queues and workers are closed!');
};
