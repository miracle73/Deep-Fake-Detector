import bcrypt from 'bcryptjs';
import mongoose, { Schema } from 'mongoose';

import logger from '../utils/logger.js';
import { AnalysisHistorySchema } from './subdocs/AnalysisHistory.js';
import { ApiAccessSchema } from './subdocs/ApiAccess.js';
import { BillingContactSchema } from './subdocs/BillingContact.js';
import { BillingHistorySchema } from './subdocs/BillingHistory.js';
import { CompanySchema } from './subdocs/Company.js';
import { PaymentMethodSchema } from './subdocs/PaymentMethod.js';
import { TeamMemberSchema } from './subdocs/TeamMember.js';
import { UsageQuotaSchema } from './subdocs/UsageQuota.js';

import type { IUser } from '../types/user.js';

const UserSchema: Schema = new Schema(
  {
    email: {
      type: String,
      required: true,
      unique: true,
      match: [
        /^\w+([\.-]?\w+)*@\w+([\.-]?\w+)*(\.\w{2,3})+$/,
        'Please add a valid email',
      ],
    },
    password: {
      type: String,
      required: function (this: IUser) {
        return !this.isGoogleUser;
      },
      minlength: 8,
      select: false,
    },
    isGoogleUser: {
      type: Boolean,
      default: false,
    },
    googleId: {
      type: String,
    },
    isEmailVerified: {
      type: Boolean,
      default: false,
    },
    stripeCustomerId: {
      type: String,
    },
    isActive: {
      type: Boolean,
      default: false,
    },
    avatar: {
      type: String,
    },
    role: {
      type: String,
      enum: ['user', 'admin'],
      default: 'user',
    },
    plan: {
      type: String,
      enum: ['SafeGuard_Free', 'SafeGuard_Pro', 'SafeGuard_Max'],
      required: true,
      default: 'SafeGuard_Free',
    },
    analysisHistory: [
      {
        type: mongoose.Types.ObjectId,
        ref: 'Analysis',
      },
    ],
    billingHistory: [BillingHistorySchema],
    agreedToTerms: {
      type: Boolean,
      required: true,
    },
    termsAgreedAt: {
      type: Date,
      required: true,
    },
    resetPasswordToken: {
      type: String,
    },
    resetPasswordExpire: {
      type: Date,
    },
    passwordChangedAt: {
      type: Date,
    },
    accessCode: {
      type: String,
      required: true,
      trim: true,
      uppercase: true,
    },

    lastLogin: {
      type: Date,
    },
    userType: {
      type: String,
      enum: ['individual', 'enterprise'],
      required: true,
    },
    firstName: {
      type: String,
      required: function (this: IUser) {
        return this.userType === 'individual';
      },
    },
    lastName: {
      type: String,
      required: function (this: IUser) {
        return this.userType === 'individual';
      },
    },
    phoneNumber: {
      type: String,
    },
    paymentMethods: [PaymentMethodSchema],
    unlimitedQuota: {
      type: Boolean,
      default: false,
    },
    usageQuota: {
      type: UsageQuotaSchema,
      required: true,
      default: () => ({
        monthlyAnalysis: 3,
        remainingAnalysis: 3,
        lastResetAt: new Date(),
        lastUsedAt: undefined,
        carryOver: false,
      }),
    },
    currentPeriodEnd: {
      type: Date,
      // index: true,
      validate: {
        validator: function (this: IUser, value: Date) {
          return !this.isActive || value > new Date();
        },
        message: 'Active subscriptions must have future end dates',
      },
    },
    cancelAtPeriodEnd: {
      type: Boolean,
      default: false,
    },
    lastPaymentDate: Date,
    nextBillingDate: Date,
    company: {
      type: CompanySchema,
      required: function (this: IUser) {
        return this.userType === 'enterprise';
      },
    },
    billingContact: {
      type: BillingContactSchema,
      required: function (this: IUser) {
        return this.userType === 'enterprise';
      },
    },
    teamMembers: [TeamMemberSchema],
    apiAccess: ApiAccessSchema,
    notifications: [
      {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'Notification',
        index: true,
      },
    ],
    emailSubscribed: {
      type: Boolean,
      default: true,
    },
    consent: {
      storeMedia: {
        type: Boolean,
        default: false,
        required: true,
      },
      updatedAt: {
        type: Date,
        default: null,
      },
    },

    sessionVersion: {
      type: Number,
      default: 0,
    },
  },
  {
    timestamps: true,
    discriminatorKey: 'userType',
  }
);

UserSchema.pre<IUser>('save', async function (next) {
  if (!this.isModified('password')) return next();

  const salt = await bcrypt.genSalt(10);
  this.password = await bcrypt.hash(this.password, salt);
  next();
});

UserSchema.pre('save', function (next) {
  if (this.isModified('currentPeriodEnd') && this.isActive) {
    if (this.currentPeriodEnd && this.currentPeriodEnd <= new Date()) {
      this.isActive = false;
      logger.warn(`Auto-deactivated expired subscription for user ${this._id}`);
    }
  }
  next();
});

UserSchema.methods.matchPassword = async function (enteredPassword: string) {
  return await bcrypt.compare(enteredPassword, this.password);
};

UserSchema.index({
  isActive: 1,
  currentPeriodEnd: 1,
});

UserSchema.index({
  cancelAtPeriodEnd: 1,
});

export default mongoose.model<IUser>('User', UserSchema);
