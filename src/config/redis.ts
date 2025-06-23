import { config } from 'dotenv';
import IORedis from 'ioredis';

import type { RedisOptions } from 'bullmq';

config();

// export const redisConnection: RedisOptions = {
//   host: process.env.REDIS_HOST || '127.0.0.1',
//   port: Number.parseInt(process.env.REDIS_PORT || '6379', 10),
// };

export const redisConnection = new IORedis(
  process.env.REDIS_URL || 'redis://127.0.0.1:6379',
  {
    enableOfflineQueue: true,
    maxRetriesPerRequest: null,
    retryStrategy: (times) => {
      if (times > 10) {
        console.log('REDIS: failed to connect arter 10 tries');
        return null;
      }

      return 3000;
    },
  }
);
