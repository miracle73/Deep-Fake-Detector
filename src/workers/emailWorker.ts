import { Worker } from 'bullmq';
import { redisConnection } from '../config/redis';
// import {
//   sendReceiptEmail,
//   sendInvoiceReminder,
//   sendFailureEmail,
// } from '../utils/email.templates.ts';

const worker = new Worker(
  'emailQueue',
  async (job) => {
    const { name, to } = job.data;

    switch (job.name) {
      case 'sendReceiptEmail':
        // await sendReceiptEmail(job.data);
        break;

      case 'upcomingInvoiceReminder':
        // await sendInvoiceReminder(job.data);
        break;

      case 'paymentFailedNotice':
        // await sendFailureEmail(job.data);
        break;

      default:
        throw new Error(`Unknown job type: ${job.name}`);
    }
  },
  { connection: redisConnection }
);

worker.on('completed', (job) => {
  console.log(`✅ Email job ${job.name} completed`);
});

worker.on('failed', (job, err) => {
  console.error(`❌ Email job ${job?.name} failed:`, err);
});
