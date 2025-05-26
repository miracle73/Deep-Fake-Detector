import { Schema } from 'mongoose';

export const ApiAccessSchema = new Schema(
  {
    enabled: { type: Boolean, default: false },
    apiKey: { type: String },
    rateLimit: { type: Number, default: 100 },
  },
  { _id: false }
);
