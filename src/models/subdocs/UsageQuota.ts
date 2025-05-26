import { Schema } from 'mongoose';

export const UsageQuotaSchema = new Schema(
  {
    monthlyAnalysis: { type: Number, default: 0 },
    remainingAnalysis: { type: Number, default: 0 },
    lastReset: { type: Date, default: Date.now },
  },
  { _id: false }
);
