import mongoose from 'mongoose';

import type { Document } from 'mongoose';

export interface ISubscription extends Document {
  userId: mongoose.Types.ObjectId;
  planId: string;
  priceId: string;
  productId: string;
  stripeSubscriptionId: string;
  status:
    | 'active'
    | 'incomplete'
    | 'incomplete_expired'
    | 'trialing'
    | 'past_due'
    | 'canceled'
    | 'unpaid';
  currentPeriodStart: Date;
  currentPeriodEnd: Date;
  cancelAtPeriodEnd: boolean;
  canceledAt: Date;
}

const subscriptionSchema = new mongoose.Schema<ISubscription>(
  {
    userId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'User',
      required: true,
    },
    planId: {
      type: String,
      required: true,
      enum: ['free', 'pro', 'max'],
    },
    priceId: {
      type: String,
      required: true,
    },
    productId: {
      type: String,
    },
    stripeSubscriptionId: {
      type: String,
      required: true,
      unique: true,
    },
    status: {
      type: String,
      enum: [
        'active',
        'incomplete',
        'incomplete_expired',
        'trialing',
        'past_due',
        'canceled',
        'unpaid',
      ],
      required: true,
    },
    currentPeriodStart: {
      type: Date,
      required: true,
    },
    currentPeriodEnd: {
      type: Date,
      required: true,
    },
    cancelAtPeriodEnd: {
      type: Boolean,
      default: false,
    },
    canceledAt: {
      type: Date,
    },
  },
  {
    timestamps: true,
  }
);

export default mongoose.model<ISubscription>(
  'Subscription',
  subscriptionSchema
);
