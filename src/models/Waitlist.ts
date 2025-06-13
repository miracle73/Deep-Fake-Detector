import mongoose from 'mongoose';
import type { IWaitlist } from '../types/waitlist.d.js';

const waitlistSchema = new mongoose.Schema<IWaitlist>(
  {
    email: {
      type: String,
      required: true,
      unique: true,
      trim: true,
      lowercase: true,
      index: true,
    },
    name: {
      type: String,
      default: null,
      trim: true,
    },
    position: {
      type: Number,
      required: true,
      index: true,
    },
    ipAddress: {
      type: String,
      required: true,
    },
    userAgent: {
      type: String,
    },
    joinedAt: {
      type: Date,
      default: Date.now,
      index: true,
    },
    status: {
      type: String,
      enum: ['active', 'invited', 'converted'],
      default: 'active',
      index: true,
    },
    invitedAt: {
      type: Date,
    },
    convertedAt: {
      type: Date,
    },
    referralCode: {
      type: Date,
      default: null,
    },
    referredBy: {
      type: String,
      default: null,
    },
  },
  { timestamps: true }
);

waitlistSchema.index({ status: 1, position: 1 });
waitlistSchema.index({ email: 1, status: 1 });

waitlistSchema.statics.getNextPosition = async function (): Promise<number> {
  const lastEntry = await this.findOne({}, {}, { sort: { position: -1 } });
  return lastEntry ? lastEntry.position + 1 : 1;
};

waitlistSchema.statics.getUserPosition = async function (
  email: string
): Promise<number | null> {
  const user = await this.findOne({ email, status: 'active' });
  if (!user) return null;

  const activeUsersAhead = await this.countDocuments({
    status: 'active',
    position: { $lt: user.position },
  });

  return activeUsersAhead + 1;
};

export default mongoose.model<IWaitlist>('Waitlist', waitlistSchema);
