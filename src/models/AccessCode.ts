import mongoose, { Schema } from 'mongoose';
import type { Document } from 'mongoose';

export interface IAccessCode extends Document {
  code: string;
  used: boolean;
  usedBy?: mongoose.Types.ObjectId;
  usedAt?: Date;
  createdAt: Date;
  expiresAt: Date;
}

const AccessCodeSchema: Schema = new Schema({
  code: {
    type: String,
    required: true,
    unique: true,
    trim: true,
    uppercase: true,
  },
  used: {
    type: Boolean,
    default: false,
  },
  usedBy: {
    type: Schema.Types.ObjectId,
    ref: 'User',
    default: null,
  },
  usedAt: {
    type: Date,
    default: null,
  },
  createdAt: {
    type: Date,
    default: Date.now,
  },
  expiresAt: {
    type: Date,
    required: true,
  },
});

AccessCodeSchema.index({ code: 1, used: 1 });
AccessCodeSchema.index({ expiresAt: 1 }, { expireAfterSeconds: 0 });

export default mongoose.model<IAccessCode>('AccessCode', AccessCodeSchema);
