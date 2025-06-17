import { emailWorker } from '../workers/emailWorker.js';
import emailQueue, { emailQueueEvent } from '../queues/emailQueue.js';
import logger from '../utils/logger.js';

export const startQueues = async () => {
  await emailQueue.waitUntilReady();
  await emailWorker.waitUntilReady();
  await emailQueueEvent.waitUntilReady();
  logger.info('Queues and workers are ready!');
};

export const stopQueues = async () => {
  await emailQueue.close();
  await emailWorker.close();
  logger.info('Queues and workers are closed!');
};
