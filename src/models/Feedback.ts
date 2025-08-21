import mongoose, { Schema } from 'mongoose';
import type { Document } from 'mongoose';

export type FeedbackType =
  | 'General Feedback'
  | 'Bug Report'
  | 'Feature Request'
  | 'Improvement';

export type FeedbackStatus = 'pending' | 'in progress' | 'resolved';

export interface IFeedback extends Document {
  type: FeedbackType;
  rating: number;
  email?: string | null;
  description: string;
  status: FeedbackStatus;
  createdAt: Date;
  updatedAt: Date;
}

const feedbackSchema = new Schema(
  {
    type: {
      type: String,
      enum: [
        'General Feedback',
        'Bug Report',
        'Feature Request',
        'Improvement',
      ],
      required: true,
    },
    rating: {
      type: Number,
      min: 1,
      max: 5,
      required: true,
    },
    email: {
      type: String,
      default: null,
    },
    description: {
      type: String,
      required: true,
    },
    status: {
      type: String,
      enum: ['pending', 'in progress', 'resolved'],
      default: 'pending',
    },
  },
  { timestamps: true }
);

feedbackSchema.index({ status: 1, createdAt: -1 });
feedbackSchema.index({ type: 1, rating: 1 });

export const Feedback = mongoose.model<IFeedback>('Feedback', feedbackSchema);
