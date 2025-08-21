import {
  createFeedbackSchema,
  feedbackQuerySchema,
  updateFeedbackSchema,
} from '../lib/schemas/feedback.schema.js';
import { feedbackService } from '../services/feedback.service.js';

import type { Request, Response } from 'express';
import type {
  CreateFeedbackInput,
  UpdateFeedbackInput,
  FeedbackQueryInput,
} from '../lib/schemas/feedback.schema.js';

export class FeedbackController {
  async createFeedback(req: Request, res: Response) {
    try {
      const validatedData: CreateFeedbackInput = createFeedbackSchema.parse(
        req.body
      );

      if (validatedData.email === '') {
        validatedData.email = undefined;
      }

      const feedback = await feedbackService.createFeedback(validatedData);
      res.status(201).json({
        success: true,
        message: 'Feedback submitted successfully',
        data: feedback,
      });
    } catch (error) {
      if (error instanceof Error) {
        res.status(400).json({
          success: false,
          message: error.message,
        });
      }
    }
  }

  async getFeedback(req: Request, res: Response) {
    try {
      const { id } = req.params;
      const feedback = await feedbackService.getFeedbackById(id);

      if (!feedback) {
        res.status(404).json({
          success: false,
          message: 'Feedback not found',
        });
        return;
      }

      res.json({
        success: true,
        message: 'Feedback retrieved successfully',
        data: feedback,
      });
    } catch (error) {
      res.status(500).json({
        success: false,
        message: 'Internal server error',
      });
    }
  }

  async getAllFeedback(req: Request, res: Response) {
    try {
      const query: FeedbackQueryInput = feedbackQuerySchema.parse(req.query);

      const result = await feedbackService.getAllFeedback({
        page: query.page,
        limit: query.limit,
        type: query.type,
        status: query.status,
        minRating: query.minRating,
        maxRating: query.maxRating,
      });

      res.json({
        success: true,
        message: 'Feedback retrieved successfully',
        pagination: {
          page: query.page,
          limit: query.limit,
          total: result.total,
          totalPages: result.totalPages,
        },
        data: result.feedback,
      });
    } catch (error) {
      res.status(400).json({
        success: false,
        message: 'Invalid query parameters',
      });
    }
  }

  async updateFeedback(req: Request, res: Response) {
    try {
      const { id } = req.params;
      const validatedData: UpdateFeedbackInput = updateFeedbackSchema.parse(
        req.body
      );

      if (validatedData.email === '') {
        validatedData.email = null;
      }

      const feedback = await feedbackService.updateFeedback(id, validatedData);

      if (!feedback) {
        res.status(404).json({
          success: false,
          message: 'Feedback not found',
        });
        return;
      }

      res.json({
        success: true,
        message: 'Feedback updated successfully',
        data: feedback,
      });
    } catch (error) {
      if (error instanceof Error) {
        res.status(400).json({
          success: false,
          message: error.message,
        });
      }
    }
  }

  async deleteFeedback(req: Request, res: Response) {
    try {
      const { id } = req.params;
      const feedback = await feedbackService.deleteFeedback(id);

      if (!feedback) {
        res.status(404).json({
          success: false,
          message: 'Feedback not found',
        });
        return;
      }

      res.json({
        success: true,
        message: 'Feedback deleted successfully',
      });
    } catch (error) {
      res.status(500).json({
        success: false,
        message: 'Internal server error',
      });
    }
  }

  async getFeedbackStats(req: Request, res: Response) {
    try {
      const stats = await feedbackService.getFeedbackStats();

      res.json({
        success: true,
        data: stats,
      });
    } catch (error) {
      res.status(500).json({
        success: false,
        message: 'Internal server error',
      });
    }
  }
}

export const feedbackController = new FeedbackController();
