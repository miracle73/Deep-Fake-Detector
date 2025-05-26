import { Schema } from 'mongoose';

export const AnalysisHistorySchema = new Schema(
  {
    analysisId: { type: String, required: true },
    date: { type: Date, default: Date.now },
    mediaType: {
      type: String,
      enum: ['image', 'video', 'audio'],
      required: true,
    },
    result: {
      type: String,
      enum: ['real', 'fake', 'inconclusive'],
      required: true,
    },
    confidence: { type: Number, required: true },
    mediaUrl: { type: String },
  },
  { _id: false }
);
