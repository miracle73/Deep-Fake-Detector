import type { Document, Types } from 'mongoose';

export interface AnalysisHistoryItem {
  analysisId: string;
  date: Date;
  mediaType: 'image' | 'video';
  result: 'real' | 'fake' | 'inconclusive';
  confidenceScore: number;
  mediaUrl?: string;
}

export interface BillingHistoryItem {
  invoiceId: string;
  date: Date;
  amount: number;
  plan: 'free' | 'pro' | 'max';
  status: 'paid' | 'pending' | 'failed';
  paymentMethod?: string;
}

export interface TeamMember {
  userId: string;
  email: string;
  role: 'admin' | 'member' | 'billing';
  joinedAt: Date;
}

export interface PaymentMethod {
  id: string;
  type: 'card' | 'paypal' | 'bank_transfer';
  lastFour?: string;
  expiry?: string;
  isDefault: boolean;
}

export interface BaseUserFields {
  email: string;
  password: string;
  isGoogleUser: boolean;
  isEmailVerified: boolean;
  stripeCustomerId?: string;
  googleId: string;
  phoneNumber?: string;
  avatar?: string;
  plan: 'free' | 'pro' | 'max';
  createdAt: Date;
  updatedAt: Date;
  agreedToTerms: boolean;
  isActive: boolean;
  termsAgreedAt: Date;
  resetPasswordToken?: string;
  resetPasswordExpire?: Date;
  passwordChangedAt?: Date;
  unlimitedQuota: boolean;
  lastLogin?: Date;
  role: string;
  usageQuota: {
    monthlyAnalysis: number;
    remainingAnalysis: number;
    lastReset: Date;
    lastResetAt: Date;
    lastUsedAt: Date;
  };
}

export interface IndividualUser extends BaseUserFields {
  firstName: string;
  lastName: string;
  userType: 'individual';
  paymentMethods?: PaymentMethod[];
}

export interface EnterpriseUser extends BaseUserFields {
  userType: 'enterprise';
  company: {
    name: string;
    website: string;
    size?: string; // '1-10', '11-50', '51-200', '201-500', '500+'
    industry?: string;
  };
  teamMembers?: TeamMember[];
  billingContact: {
    name: string;
    email: string;
    phone: string;
  };
  apiAccess: {
    enabled: boolean;
    apiKey?: string;
    rateLimit: number;
  };
}

export type IUser = Document &
  (IndividualUser | EnterpriseUser) & {
    _id: Types.ObjectId;
    analysisHistory: AnalysisHistoryItem[];
    billingHistory: BillingHistoryItem[];
    createdAt: Date;
    updatedAt: Date;
  };

export interface BaseUserResponse {
  id: string;
  email: string;
  userType: 'individual' | 'enterprise';
  plan: 'free' | 'pro' | 'max';
}

export interface IndividualUserResponse extends BaseUserResponse {
  userType: 'individual';
  firstName: string;
  lastName: string;
}

export interface EnterpriseUserResponse extends BaseUserResponse {
  userType: 'enterprise';
  company: {
    name: string;
    website: string;
    size?: string;
    industry?: string;
  };
  billingContact: {
    name: string;
    email: string;
    phone: string;
  };
}

export type UserResponse = IndividualUserResponse | EnterpriseUserResponse;

export interface AuthResponse {
  success: boolean;
  token: string;
  user: UserResponse;
}

export const formatUserResponse = (user: IUser): UserResponse => {
  const baseResponse: BaseUserResponse = {
    id: user._id.toString(),
    email: user.email,
    userType: user.userType,
    plan: user.plan,
  };

  if (user.userType === 'individual') {
    return {
      ...baseResponse,
      userType: 'individual',
      firstName: (user as IndividualUser).firstName,
      lastName: (user as IndividualUser).lastName,
    };
  }
  return {
    ...baseResponse,
    userType: 'enterprise',
    company: (user as EnterpriseUser).company,
    billingContact: (user as EnterpriseUser).billingContact,
  };
};

export interface GoogleTempUser {
  email: string;
  googleId: string;
  firstName: string;
  lastName: string;
}
