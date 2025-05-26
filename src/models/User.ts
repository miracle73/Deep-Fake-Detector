import bcrypt from 'bcryptjs';
import mongoose, { Schema } from 'mongoose';

import { AnalysisHistorySchema } from './subdocs/AnalysisHistory';
import { BillingHistorySchema } from './subdocs/BillingHistory';
import { PaymentMethodSchema } from './subdocs/PaymentMethod';

import type { IUser } from '../types/user';
import { ApiAccessSchema } from './subdocs/ApiAccess';
import { BillingContactSchema } from './subdocs/BillingContact';
import { TeamMemberSchema } from './subdocs/TeamMember';
import { CompanySchema } from './subdocs/Company';
import { UsageQuotaSchema } from './subdocs/UsageQuota';

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
    isGoogleUser: { type: Boolean, default: false },
    isEmailVerified: { type: Boolean, default: false },
    firstName: { type: String, required: true },
    lastName: { type: String, required: true },
    phoneNumber: { type: String },
    avatar: { type: String },
    plan: {
      type: String,
      enum: ['free', 'pro', 'max'],
      default: 'free',
    },
    analysisHistory: [AnalysisHistorySchema],
    billingHistory: [BillingHistorySchema],
    agreedToTerms: { type: Boolean, required: true },
    termsAgreedAt: { type: Date, required: true },
    resetPasswordToken: { type: String },
    resetPasswordExpire: { type: Date },
    passwordChangedAt: { type: Date },
    lastLogin: { type: Date },
    userType: {
      type: String,
      enum: ['individual', 'enterprise'],
      required: true,
    },
    // Individual user fields
    paymentMethods: [PaymentMethodSchema],
    usageQuota: UsageQuotaSchema,
    // Enterprise user fields
    company: CompanySchema,
    billingContact: BillingContactSchema,
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
