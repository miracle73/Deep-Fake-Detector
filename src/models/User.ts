import bcrypt from 'bcryptjs';
import mongoose, { Schema } from 'mongoose';

import { AnalysisHistorySchema } from './subdocs/AnalysisHistory.js';
import { BillingHistorySchema } from './subdocs/BillingHistory.js';
import { PaymentMethodSchema } from './subdocs/PaymentMethod.js';

import type { IUser } from '../types/user.js';
import { ApiAccessSchema } from './subdocs/ApiAccess.js';
import { BillingContactSchema } from './subdocs/BillingContact.js';
import { TeamMemberSchema } from './subdocs/TeamMember.js';
import { CompanySchema } from './subdocs/Company.js';
import { UsageQuotaSchema } from './subdocs/UsageQuota.js';

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
      enum: ['free', 'pro', 'max'],
      default: 'free',
    },
    analysisHistory: [AnalysisHistorySchema],
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
        lastReset: new Date(),
        lastResetAt: new Date(),
        lastUsedAt: undefined,
      }),
    },
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

UserSchema.methods.matchPassword = async function (enteredPassword: string) {
  return await bcrypt.compare(enteredPassword, this.password);
};

export default mongoose.model<IUser>('User', UserSchema);
