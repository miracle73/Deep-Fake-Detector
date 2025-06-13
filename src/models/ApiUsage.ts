import mongoose, { Schema } from 'mongoose';
import type { Document } from 'mongoose';

export interface IApiUsage extends Document {
  userId: mongoose.Types.ObjectId;
  endpoint: string;
  method: string;
  timestamp: Date;
  responseTime: number;
  statusCode: number;
  ipAddress: string;
}

const apiUsageSchema = new Schema<IApiUsage>({
  userId: {
    type: Schema.Types.ObjectId,
    ref: 'User',
    required: true,
  },
  endpoint: {
    type: String,
    required: true,
  },
  method: {
    type: String,
    required: true,
  },
  timestamp: {
    type: Date,
    default: Date.now,
  },
  responseTime: {
    type: Number,
    required: true,
  },
  statusCode: {
    type: Number,
    required: true,
  },
  ipAddress: {
    type: String,
    required: true,
  },
});

export default mongoose.model<IApiUsage>('ApiUsage', apiUsageSchema);
