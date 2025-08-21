import Redis from 'ioredis';

class RedisService {
  private client: Redis;

  constructor() {
    this.client = new Redis(process.env.REDIS_URL || 'redis://localhost:6379');
  }

  // biome-ignore lint/suspicious/noExplicitAny: <explanation>
  async set(key: string, value: any, expiration?: number): Promise<void> {
    const stringValue = JSON.stringify(value);
    if (expiration) {
      await this.client.setex(key, expiration, stringValue);
    } else {
      await this.client.set(key, stringValue);
    }
  }

  async get<T>(key: string): Promise<T | null> {
    const value = await this.client.get(key);
    return value ? JSON.parse(value) : null;
  }

  async del(key: string): Promise<void> {
    await this.client.del(key);
  }

  async clearPattern(pattern: string): Promise<void> {
    const keys = await this.client.keys(pattern);
    if (keys.length > 0) {
      await this.client.del(...keys);
    }
  }

  async invalidateFeedbackCache(): Promise<void> {
    await this.clearPattern('feedback:*');
    await this.clearPattern('feedback_stats:*');
  }
}

export const redisService = new RedisService();
