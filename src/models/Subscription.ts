import mongoose, { Document, Schema } from 'mongoose';

export interface ISubscription extends Document {
  userId: mongoose.Types.ObjectId;
  planId: string;
  stripeSubscriptionId: string;
  status: 'active' | 'canceled' | 'past_due';
  currentPeriodStart: Date;
  currentPeriodEnd: Date;
  cancelAtPeriodEnd: boolean;
}

const subscriptionSchema = new Schema<ISubscription>(
  {
    userId: {
      type: Schema.Types.ObjectId,
      ref: 'User',
      required: true,
    },
    planId: {
      type: String,
      required: true,
      enum: ['free', 'pro', 'max'],
    },
    stripeSubscriptionId: {
      type: String,
      required: true,
      unique: true,
    },
    status: {
      type: String,
      required: true,
      enum: ['active', 'canceled', 'past_due'],
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
  },
  {
    timestamps: true,
  }
);

export default mongoose.model<ISubscription>(
  'Subscription',
  subscriptionSchema
);
