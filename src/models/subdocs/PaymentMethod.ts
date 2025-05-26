import { Schema } from 'mongoose';

export const PaymentMethodSchema = new Schema(
  {
    id: { type: String, required: true },
    type: {
      type: String,
      enum: ['card', 'paypal', 'bank_transfer'],
      required: true,
    },
    lastFour: { type: String },
    expiry: { type: String },
    isDefault: { type: Boolean, default: false },
  },
  { _id: false }
);
