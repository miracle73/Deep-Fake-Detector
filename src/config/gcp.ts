import { Logging } from '@google-cloud/logging';
import { MetricServiceClient } from '@google-cloud/monitoring';
import { config } from 'dotenv';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

config();

config({ path: path.resolve(process.cwd(), '.env') });

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// const __dirname = path.resolve();
process.env.GOOGLE_APPLICATION_CREDENTIALS = path.resolve(
  __dirname,
  'gcp-key.json'
);

const logging = new Logging();
const metricClient = new MetricServiceClient();

export { logging, metricClient };
