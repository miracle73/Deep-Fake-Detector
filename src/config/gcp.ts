import { Logging } from '@google-cloud/logging';
import { MetricServiceClient } from '@google-cloud/monitoring';

const logging = new Logging();
const metricClient = new MetricServiceClient();

export { logging, metricClient };
// This module initializes Google Cloud Logging and Monitoring clients.
// It exports the logging and metricClient instances for use in other parts of the application.
// This allows the application to log messages and metrics to Google Cloud services.
// The Logging client is used for structured logging, while the MetricServiceClient is used for custom metrics.
