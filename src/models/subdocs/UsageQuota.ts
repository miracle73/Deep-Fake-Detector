import { Schema } from 'mongoose';

export const UsageQuotaSchema = new Schema(
  {
    monthlyAnalysis: {
      type: Number,
      required: true,
      default: 3,
    },
    remainingAnalysis: {
      type: Number,
      default: 0,
    },
    lastReset: {
      type: Date,
      default: Date.now,
    },
  },
  { _id: false }
);
