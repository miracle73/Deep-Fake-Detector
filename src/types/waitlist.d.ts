export interface IWaitlist extends Document {
  email: string;
  position: number;
  name?: string;
  ipAddress: string;
  userAgent?: string;
  joinedAt: Date;
  invitedAt?: Date;
  convertedAt?: Date;
  status: 'active' | 'invited' | 'converted';
  referralCode?: Date;
  referredBy?: string;
}

export interface WaitlistSignupData {
  email: string;
  ipAddress: string;
  userAgent?: string;
}

export interface WaitlistSignupResult {
  email: string;
  position: number;
  signupDate: Date;
  estimatedWaitTime: string;
  totalSignups: number;
}

export interface WaitlistStatusResult {
  email: string;
  currentPosition: number;
  originalPosition: number;
  signupDate: Date;
  status: string;
  estimatedWaitTime: string;
  totalActive: number;
}
