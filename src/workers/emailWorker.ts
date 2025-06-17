import { Worker } from 'bullmq';

import { redisConnection } from '../config/redis';
import { sendEmail } from '../services/emailService.js';
import logger from '../utils/logger.js';

import type { EmailJob } from '../types/email.d.js';

logger.info('🚀 Email worker is running...');

const mainWorkerOptions = {
  connection: redisConnection,
  concurrency: 5,
  removeOnFail: { count: 0 },
};

export const emailWorker = new Worker(
  'emailQueue',
  async (job) => {
    logger.info(job.name);
    // logger.info(job.data);
    mailUser(job);
  },
  mainWorkerOptions
);

emailWorker.on('error', (err) => {
  logger.error(`Error processing job: ${err}`);
});

emailWorker.on('completed', (job) => {
  logger.info(`✅ Email job ${job.name} completed`);
});

emailWorker.on('failed', (job, err) => {
  logger.error(`❌ Email job ${job?.name} failed:`, err);
});

const mailUser = async (job: EmailJob) => {
  const { to, subject, html } = job.data;
  await sendEmail({
    to,
    subject,
    html,
  });

  return;
};
