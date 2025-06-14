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
  firstName: string;
  stripeCustomerId?: string;
  lastName: string;
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
  lastLogin?: Date;
  role: string;
}

export interface IndividualUser extends BaseUserFields {
  userType: 'individual';
  paymentMethods?: PaymentMethod[];
  usageQuota: {
    monthlyAnalysis: number;
    remainingAnalysis: number;
    lastReset: Date;
  };
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
