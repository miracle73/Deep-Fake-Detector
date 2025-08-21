import { z } from 'zod';

export const feedbackTypeSchema = z.enum([
  'General Feedback',
  'Bug Report',
  'Feature Request',
  'Improvement',
]);
export const feedbackStatusSchema = z.enum([
  'pending',
  'in progress',
  'resolved',
]);

export const createFeedbackSchema = z.object({
  type: feedbackTypeSchema,
  rating: z.number().min(1).max(5),
  email: z.string().email().optional().or(z.literal('')),
  description: z.string().min(10, 'Description must be at least 10 characters'),
  status: feedbackStatusSchema.optional(),
});

export const updateFeedbackSchema = z.object({
  type: feedbackTypeSchema,
  rating: z.number().min(1).max(5),
  email: z.string().email().optional().or(z.literal('')).nullable(),
  description: z.string().min(10, 'Description must be at least 10 characters'),
  status: feedbackStatusSchema.default('pending'),
});

export const feedbackQuerySchema = z.object({
  page: z
    .string()
    .optional()
    .transform((val) => Number.parseInt(val || '1')),
  limit: z
    .string()
    .optional()
    .transform((val) => Number.parseInt(val || '10')),
  type: feedbackTypeSchema.optional(),
  status: feedbackStatusSchema.optional(),
  minRating: z
    .string()
    .optional()
    .transform((val) => Number.parseInt(val || '1')),
  maxRating: z
    .string()
    .optional()
    .transform((val) => Number.parseInt(val || '5')),
});

export type CreateFeedbackInput = z.infer<typeof createFeedbackSchema>;
export type UpdateFeedbackInput = z.infer<typeof updateFeedbackSchema>;
export type FeedbackQueryInput = z.infer<typeof feedbackQuerySchema>;
