import { logging } from '../../config/gcp.js';
import logger from '../logger.js';

const log = logging.log('safeguard-media-log');

export const logInfo = async (message: string, metadata = {}) => {
  const entry = log.entry(
    { resource: { type: 'global' }, ...metadata },
    message
  );
  await log.write(entry);

  logger.info(`Logged info: ${message}`, metadata);
};

// biome-ignore lint/suspicious/noExplicitAny: <explanation>
export const logError = async (message: string, error: any) => {
  const entry = log.entry(
    { resource: { type: 'global' }, severity: 'ERROR' },
    {
      message,
      error: error instanceof Error ? error.message : error,
      stack: error instanceof Error ? error.stack : undefined,
      timestamp: new Date().toISOString(),
    }
  );
  await log.write(entry);

  logger.error(`Logged error: ${message}`, {
    error: error instanceof Error ? error.message : error,
    stack: error instanceof Error ? error.stack : undefined,
    timestamp: new Date().toISOString(),
  });
};
