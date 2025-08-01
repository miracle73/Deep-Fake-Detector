import { Logging } from '@google-cloud/logging';
import { MetricServiceClient } from '@google-cloud/monitoring';
import { config } from 'dotenv';

config();

if (!process.env.GOOGLE_APPLICATION_CREDENTIALS) {
  throw new Error('GOOGLE_APPLICATION_CREDENTIALS env var is required');
}

console.log(
  '✅ GOOGLE_APPLICATION_CREDENTIALS loaded:',
  process.env.GOOGLE_APPLICATION_CREDENTIALS
);

const logging = new Logging();
const metricClient = new MetricServiceClient();

console.log('✅ Google Cloud Logging and Monitoring clients initialized');

export { logging, metricClient };
