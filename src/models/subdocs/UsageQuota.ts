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
      required: true,
      default: 3,
    },
    lastReset: {
      type: Date,
      default: Date.now,
    },
    lastUsedAt: {
      type: Date,
    },
    lastResetAt: {
      type: Date,
      default: Date.now,
    },
  },
  { _id: false }
);
