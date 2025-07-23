import mongoose, { Schema } from 'mongoose';

const WebhookEventSchema = new Schema(
  {
    eventId: { type: String, required: true, unique: true },
    type: { type: String, required: true },
    processedAt: { type: Date, default: Date.now },
    userId: { type: Schema.Types.ObjectId, ref: 'User' },
    payload: { type: Schema.Types.Mixed },
  },
  { timestamps: true }
);

WebhookEventSchema.index(
  { createdAt: 1 },
  { expireAfterSeconds: 30 * 24 * 60 * 60 }
); // auto-delete after 30 days

const WebhookEvent = mongoose.model('WebhookEvent', WebhookEventSchema);
export type IWebhookEvent = {
  eventId: string;
  type: string;
  processedAt?: Date;
  userId?: mongoose.Types.ObjectId;
  // biome-ignore lint/suspicious/noExplicitAny: <explanation>
  payload?: any;
};
export default WebhookEvent;
