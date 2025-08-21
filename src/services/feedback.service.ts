import { Feedback } from '../models/Feedback.js';
import { redisService } from './redis.service.js';

import type {
  IFeedback,
  FeedbackType,
  FeedbackStatus,
} from '../models/Feedback.js';

export class FeedbackService {
  private readonly CACHE_TTL = 300; // 5 mins

  async createFeedback(data: Partial<IFeedback>): Promise<IFeedback> {
    const feedback = new Feedback(data);
    await feedback.save();
    await redisService.invalidateFeedbackCache();
    return feedback;
  }

  async getFeedbackById(id: string): Promise<IFeedback | null> {
    const cacheKey = `feedback:${id}`;
    const cached = await redisService.get<IFeedback>(cacheKey);
    if (cached) return cached;

    const feedback = await Feedback.findById(id);
    if (feedback) {
      await redisService.set(cacheKey, feedback, this.CACHE_TTL);
    }
    return feedback;
  }

  async getAllFeedback(query: {
    page: number;
    limit: number;
    type?: FeedbackType;
    status?: FeedbackStatus;
    minRating?: number;
    maxRating?: number;
  }): Promise<{ feedback: IFeedback[]; total: number; totalPages: number }> {
    const { page, limit, type, status, minRating = 1, maxRating = 5 } = query;
    const skip = (page - 1) * limit;

    // biome-ignore lint/suspicious/noExplicitAny: <explanation>
    const filter: any = {
      rating: { $gte: minRating, $lte: maxRating },
    };

    if (type) filter.type = type;
    if (status) filter.status = status;

    const cacheKey = `feedback:list:${JSON.stringify({ page, limit, filter })}`;
    const cached = await redisService.get<{
      feedback: IFeedback[];
      total: number;
      totalPages: number;
    }>(cacheKey);
    if (cached) return cached;

    const [feedback, total] = await Promise.all([
      Feedback.find(filter).sort({ createdAt: -1 }).skip(skip).limit(limit),
      Feedback.countDocuments(filter),
    ]);

    const totalPages = Math.ceil(total / limit);
    const result = { feedback, total, totalPages };

    await redisService.set(cacheKey, result, this.CACHE_TTL);
    return result;
  }

  async updateFeedback(
    id: string,
    data: Partial<IFeedback>
  ): Promise<IFeedback | null> {
    const feedback = await Feedback.findByIdAndUpdate(
      id,
      { ...data, updatedAt: new Date() },
      { new: true, runValidators: true }
    );

    if (feedback) {
      await redisService.invalidateFeedbackCache();
    }
    return feedback;
  }

  async deleteFeedback(id: string): Promise<IFeedback | null> {
    const feedback = await Feedback.findByIdAndDelete(id);
    if (feedback) {
      await redisService.invalidateFeedbackCache();
    }
    return feedback;
  }

  async getFeedbackStats(): Promise<{
    total: number;
    pending: number;
    inProgress: number;
    resolved: number;
    averageRating: number;
    ratingDistribution: { [key: number]: number };
    typeDistribution: { [key: string]: number };
  }> {
    const cacheKey = 'feedback_stats:summary';
    // biome-ignore lint/suspicious/noExplicitAny: <explanation>
    const cached = await redisService.get<any>(cacheKey);
    if (cached) return cached;

    const stats = await Feedback.aggregate([
      {
        $facet: {
          total: [{ $count: 'count' }],
          statusCounts: [{ $group: { _id: '$status', count: { $sum: 1 } } }],
          averageRating: [
            { $group: { _id: null, average: { $avg: '$rating' } } },
          ],
          ratingDistribution: [
            { $group: { _id: '$rating', count: { $sum: 1 } } },
          ],
          typeDistribution: [{ $group: { _id: '$type', count: { $sum: 1 } } }],
        },
      },
    ]);

    const result = {
      total: stats[0].total[0]?.count || 0,
      pending:
        // biome-ignore lint/suspicious/noExplicitAny: <explanation>
        stats[0].statusCounts.find((s: any) => s._id === 'pending')?.count || 0,
      inProgress:
        // biome-ignore lint/suspicious/noExplicitAny: <explanation>
        stats[0].statusCounts.find((s: any) => s._id === 'in progress')
          ?.count || 0,
      resolved:
        // biome-ignore lint/suspicious/noExplicitAny: <explanation>
        stats[0].statusCounts.find((s: any) => s._id === 'resolved')?.count ||
        0,
      averageRating: stats[0].averageRating[0]?.average || 0,
      ratingDistribution: stats[0].ratingDistribution.reduce(
        // biome-ignore lint/suspicious/noExplicitAny: <explanation>
        (acc: any, curr: any) => {
          acc[curr._id] = curr.count;
          return acc;
        },
        {}
      ),
      typeDistribution: stats[0].typeDistribution.reduce(
        // biome-ignore lint/suspicious/noExplicitAny: <explanation>
        (acc: any, curr: any) => {
          acc[curr._id] = curr.count;
          return acc;
        },
        {}
      ),
    };

    await redisService.set(cacheKey, result, this.CACHE_TTL);
    return result;
  }
}

export const feedbackService = new FeedbackService();
