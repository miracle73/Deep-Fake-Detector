import mongoose, { Schema } from 'mongoose';
import type { Document } from 'mongoose';

export interface IPaymentHistory extends Document {
  userId: mongoose.Types.ObjectId;
  stripePaymentIntentId: string;
  amount: number;
  currency: string;
  status: 'succeeded' | 'failed' | 'pending';
  paymentMethod: string;
}

const paymentHistorySchema = new Schema<IPaymentHistory>(
  {
    userId: {
      type: Schema.Types.ObjectId,
      ref: 'User',
      required: true,
    },
    stripePaymentIntentId: {
      type: String,
      required: true,
      unique: true,
    },
    amount: {
      type: Number,
      required: true,
    },
    currency: {
      type: String,
      required: true,
      default: 'usd',
    },
    status: {
      type: String,
      required: true,
      enum: ['succeeded', 'failed', 'pending'],
    },
    paymentMethod: {
      type: String,
      required: true,
    },
  },
  {
    timestamps: true,
  }
);

export default mongoose.model<IPaymentHistory>(
  'PaymentHistory',
  paymentHistorySchema
);
