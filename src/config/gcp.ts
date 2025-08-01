import { Logging } from '@google-cloud/logging';
import { MetricServiceClient } from '@google-cloud/monitoring';
import { config } from 'dotenv';

config();

if (!process.env.GOOGLE_CREDENTIALS_JSON) {
  throw new Error('GOOGLE_CREDENTIALS_JSON environment variable is required');
}

const credentials = JSON.parse(process.env.GOOGLE_CREDENTIALS_JSON);

const logging = new Logging({
  projectId: credentials.project_id,
  credentials,
});

const metricClient = new MetricServiceClient({
  projectId: credentials.project_id,
  credentials,
});

export { logging, metricClient };
