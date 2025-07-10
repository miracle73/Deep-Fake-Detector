import mongoose from 'mongoose';

const NotificationSchema = new mongoose.Schema(
  {
    userId: {
      type: mongoose.Schema.Types.ObjectId,
      required: true,
      ref: 'User',
      index: true,
    },
    type: {
      type: String,
      enum: [
        'media_verified',
        'media_flagged',
        'system',
        'credential_verified',
        'credential_denied',
        'transaction',
        'account',
        'promotional',
      ],
      required: true,
      index: true,
    },
    title: {
      type: String,
      required: true,
      maxlength: 100,
    },
    message: {
      type: String,
      required: true,
      maxlength: 500,
    },
    link: {
      type: String,
      validate: {
        // biome-ignore lint/suspicious/noExplicitAny: <explanation>
        validator: (v: any) => {
          return v === undefined || /^(https?|app):\/\/.+/i.test(v);
        },
        message: 'Link must be a valid URL or app deep link',
      },
    },
    read: {
      type: Boolean,
      default: false,
      index: true,
    },
    metadata: {
      type: mongoose.Schema.Types.Mixed,
      default: {},
    },
    expiresAt: {
      type: Date,
      index: { expires: '90d' },
    },
  },
  {
    timestamps: true,
    toJSON: { virtuals: true },
    toObject: { virtuals: true },
  }
);

NotificationSchema.index({ userId: 1, read: 1 });
NotificationSchema.index({ userId: 1, type: 1 });

NotificationSchema.statics.findUnreadByUser = function (userId) {
  return this.find({ userId, read: false }).sort({ createdAt: -1 });
};

export const Notification = mongoose.model('Notification', NotificationSchema);
