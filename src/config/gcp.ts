import { Logging } from '@google-cloud/logging';
import { MetricServiceClient } from '@google-cloud/monitoring';
import { config } from 'dotenv';
import path from 'node:path';

config();

const __dirname = path.resolve();
process.env.GOOGLE_APPLICATION_CREDENTIALS = path.resolve(
  __dirname,
  './src/config/gcp-key.json'
);

const logging = new Logging();
const metricClient = new MetricServiceClient();

export { logging, metricClient };
