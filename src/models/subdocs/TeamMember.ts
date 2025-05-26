import { Schema } from 'mongoose';

export const TeamMemberSchema = new Schema(
  {
    userId: { type: Schema.Types.ObjectId, ref: 'User', required: true },
    email: { type: String, required: true },
    role: {
      type: String,
      enum: ['admin', 'member', 'billing'],
      required: true,
    },
    joinedAt: { type: Date, default: Date.now },
  },
  { _id: false }
);
