import { Schema } from 'mongoose';

export const BillingHistorySchema = new Schema(
  {
    invoiceId: {
      type: String,
      required: true,
    },
    date: {
      type: Date,
      default: Date.now,
    },
    amount: {
      type: Number,
      required: true,
    },
    plan: {
      type: String,
      enum: ['SafeGuard_Free', 'SafeGuard_Pro', 'SafeGuard_Max'],
      required: true,
    },
    status: {
      type: String,
      enum: ['paid', 'pending', 'failed'],
      required: true,
    },
    paymentMethod: { type: String },
  },
  { _id: false }
);
