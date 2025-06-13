import type { IUser, IndividualUser, EnterpriseUser } from '../types/user.js';

interface BaseUserResponse {
  id: string;
  email: string;
  userType: 'individual' | 'enterprise';
  plan: 'free' | 'pro' | 'max';
}

interface IndividualUserResponse extends BaseUserResponse {
  userType: 'individual';
  firstName: string;
  lastName: string;
}

interface EnterpriseUserResponse extends BaseUserResponse {
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
    const individualUser = user as IndividualUser;
    return {
      ...baseResponse,
      userType: 'individual',
      firstName: individualUser.firstName,
      lastName: individualUser.lastName,
    };
  }

  const enterpriseUser = user as EnterpriseUser;
  return {
    ...baseResponse,
    userType: 'enterprise',
    company: enterpriseUser.company,
    billingContact: enterpriseUser.billingContact,
  };
};
