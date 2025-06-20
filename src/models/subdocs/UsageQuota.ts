import { Schema } from 'mongoose';

import { getMonthlyResetDate } from '../../utils/dateUtils.js';

export const UsageQuotaSchema = new Schema(
  {
    monthlyAnalysis: {
      type: Number,
      required: true,
      // biome-ignore lint/suspicious/noExplicitAny: <explanation>
      default: function (this: any) {
        return this.parent().plan === 'pro' ? 100 : 3;
      },
    },
    remainingAnalysis: {
      type: Number,
      min: 0,
      required: true,
      // biome-ignore lint/suspicious/noExplicitAny: <explanation>
      default: function (this: any) {
        return this.monthlyAnalysis;
      },
    },
    lastResetAt: {
      type: Date,
      default: () => {
        return getMonthlyResetDate();
      },
    },
    lastUsedAt: Date,
    carryOver: {
      type: Boolean,
      default: false,
    },
  },
  { _id: false }
);
