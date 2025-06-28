import { logging } from '../../config/gcp.js';
import logger from '../logger.js';

const log = logging.log('safeguard-media-log');

interface LogPayload {
  message: string;
  context?: Record<string, unknown>;
  userId?: string;
  requestId?: string;
  endpoint?: string;
  metadata?: Record<string, unknown>;
}

export const cloudLogger = {
  info: async ({ message, ...metadata }: LogPayload) => {
    const entry = log.entry(
      {
        resource: { type: 'global' },
        labels: {
          service: 'deepfake-detector',
          ...(metadata.userId && { user: metadata.userId }),
        },
        ...metadata.context,
      },
      message
    );

    await log.write(entry).catch((e) => {
      logger.error('Failed to write to Cloud Logging', e);
    });

    logger.info(message, metadata);
  },

  error: async ({
    message,
    error,
    ...metadata
  }: LogPayload & { error: unknown }) => {
    const errorData = {
      message: error instanceof Error ? error.message : String(error),
      stack: error instanceof Error ? error.stack : undefined,
      name: error instanceof Error ? error.name : undefined,
    };

    const entry = log.entry(
      {
        resource: { type: 'global' },
        severity: 'ERROR',
        labels: {
          service: 'deepfake-detector',
          error_type: errorData.name ? errorData.name : 'UnknownError',
          ...(metadata.userId && { user: metadata.userId }),
        },
        ...metadata.context,
      },
      {
        ...errorData,
        ...metadata,
      }
    );

    await log.write(entry).catch((e) => {
      logger.error('Failed to write error to Cloud Logging', e);
    });

    logger.error(message, { ...metadata, error: errorData });
  },
};

// export const logInfo = async (message: string, metadata = {}) => {
//   const entry = log.entry(
//     { resource: { type: 'global' }, ...metadata },
//     message
//   );
//   await log.write(entry);

//   logger.info(`Logged info: ${message}`, metadata);
// };

// // biome-ignore lint/suspicious/noExplicitAny: <explanation>
// export const logError = async (message: string, error: any) => {
//   const entry = log.entry(
//     { resource: { type: 'global' }, severity: 'ERROR' },
//     {
//       message,
//       error: error instanceof Error ? error.message : error,
//       stack: error instanceof Error ? error.stack : undefined,
//       timestamp: new Date().toISOString(),
//     }
//   );
//   await log.write(entry);

//   logger.error(`Logged error: ${message}`, {
//     error: error instanceof Error ? error.message : error,
//     stack: error instanceof Error ? error.stack : undefined,
//     timestamp: new Date().toISOString(),
//   });
// };
