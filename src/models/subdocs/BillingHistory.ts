import { Schema } from 'mongoose';

export const BillingHistorySchema = new Schema(
  {
    invoiceId: { type: String, required: true },
    date: { type: Date, default: Date.now },
    amount: { type: Number, required: true },
    plan: { type: String, enum: ['free', 'pro', 'max'], required: true },
    status: {
      type: String,
      enum: ['paid', 'pending', 'failed'],
      required: true,
    },
    paymentMethod: { type: String },
  },
  { _id: false }
);
