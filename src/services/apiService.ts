import { createApi, fetchBaseQuery } from "@reduxjs/toolkit/query/react";

interface RegisterRequest {
  email: string;
  password: string;
  firstName: string;
  lastName: string;
  agreedToTerms: boolean;
  userType: string;
}

interface RootState {
  auth: {
    isAuthenticated: boolean;
    token: string | null;
    expires_in: string;
    lastActivity: number;
  };
}

interface LoginRequest {
  email: string;
  password: string;
}

interface ForgotPasswordRequest {
  email: string;
}

interface GoogleLoginRequest {
  idToken: {
    access_token: string;
  };
  agreedToTerms: boolean;
  userType: string;
}

interface BillingContact {
  name: string;
  email: string;
  phone: string;
}

interface Company {
  name: string;
  website: string;
  size: string;
  industry: string;
}

interface EnterpriseRegisterRequest {
  email: string;
  password: string;
  agreedToTerms: boolean;
  userType: string;
  billingContact: BillingContact;
  company: Company;
}

interface UpdateUserRequest {
  email?: string;
  password?: string;
  firstName?: string;
  lastName?: string;
}

interface UserResponse {
  id: string;
  email: string;
  userType: string;
  plan: string;
  firstName?: string;
  lastName?: string;
  company?: Company;
  billingContact?: BillingContact;
}

interface AnalysisHistory {
  id: string;
  type: string;
  result: string;
  createdAt: string;
  fileName?: string;
}

interface BillingHistory {
  id: string;
  amount: number;
  currency: string;
  status: string;
  createdAt: string;
  description?: string;
}

interface PaymentMethod {
  id: string;
  type: string;
  last4?: string;
  brand?: string;
  isDefault: boolean;
}

interface TeamMember {
  id: string;
  email: string;
  role: string;
  firstName?: string;
  lastName?: string;
  invitedAt: string;
  status: string;
}

interface DetailedUser {
  _id: string;
  email: string;
  isGoogleUser: boolean;
  isEmailVerified: boolean;
  stripeCustomerId: string;
  isActive: boolean;
  role: string;
  plan: string;
  agreedToTerms: boolean;
  termsAgreedAt: string;
  userType: string;
  firstName: string;
  lastName: string;
  analysisHistory: AnalysisHistory[];
  billingHistory: BillingHistory[];
  paymentMethods: PaymentMethod[];
  teamMembers: TeamMember[];
  createdAt: string;
  updatedAt: string;
  __v: number;
  lastLogin?: string;
  resetPasswordExpire?: string;
  resetPasswordToken?: string;
  company?: Company;
  billingContact?: BillingContact;
}

interface LoginResponse {
  success: boolean;
  token: string;
  user: UserResponse;
}

interface RegisterResponse {
  success: boolean;
  token: string;
  user: UserResponse;
}

interface ForgotPasswordResponse {
  success: boolean;
  code: number;
  message: string;
}

interface ResetPasswordResponse {
  success: boolean;
  message?: string;
}

interface GoogleLoginResponse {
  success: boolean;
  token: string;
  user: UserResponse;
}

interface VerifyEmailResponse {
  success: boolean;
  message?: string;
}

interface EnterpriseRegisterResponse {
  success: boolean;
  token: string;
  user: UserResponse;
}

interface GetUserResponse {
  success: boolean;
  message: string;
  data: {
    user: DetailedUser;
  };
}

interface UpdateUserResponse {
  success: boolean;
  message: string;
  data: {
    updatedUser: DetailedUser;
  };
}

interface DeleteUserResponse {
  success: boolean;
  message: string;
  data: {
    deletedUser: DetailedUser;
  };
}

export const apiService = createApi({
  reducerPath: "apiService",
  baseQuery: fetchBaseQuery({
    baseUrl:
      "https://deepfake-detector-655166731472.us-central1.run.app/api/v1",
    prepareHeaders: (headers, { getState, endpoint }) => {
      // List of endpoints that should NOT include the token
      const publicEndpoints = [
        "register",
        "login",
        "forgotPassword",
        "resetPassword",
        "googleLogin",
        "verifyEmail",
        "registerEnterprise",
      ];

      // Only add token if it's not a public endpoint
      if (!publicEndpoints.includes(endpoint)) {
        const token = (getState() as RootState).auth?.token;
        if (token) {
          headers.set("Authorization", `Bearer ${token}`);
        }
      }

      return headers;
    },
  }),
  endpoints: (builder) => ({
    register: builder.mutation<RegisterResponse, RegisterRequest>({
      query: ({
        email,
        password,
        firstName,
        lastName,
        agreedToTerms,
        userType,
      }) => ({
        url: "auth/register",
        method: "POST",
        body: { email, password, firstName, lastName, agreedToTerms, userType },
      }),
    }),
    login: builder.mutation<LoginResponse, LoginRequest>({
      query: ({ email, password }) => ({
        url: "auth/login",
        method: "POST",
        body: { email, password },
      }),
    }),
    forgotPassword: builder.mutation<
      ForgotPasswordResponse,
      ForgotPasswordRequest
    >({
      query: ({ email }) => ({
        url: "auth/forgot-password",
        method: "POST",
        body: { email },
      }),
    }),
    resetPassword: builder.mutation<
      ResetPasswordResponse,
      { token: string; password: string }
    >({
      query: ({ token, password }) => ({
        url: `auth/reset-password/${token}`,
        method: "POST",
        body: { password },
      }),
    }),
    googleLogin: builder.mutation<GoogleLoginResponse, GoogleLoginRequest>({
      query: ({ idToken, agreedToTerms, userType }) => ({
        url: "auth/google",
        method: "POST",
        body: { idToken, agreedToTerms, userType },
      }),
    }),
    verifyEmail: builder.query<VerifyEmailResponse, string>({
      query: (token) => ({
        url: `auth/verify-email?token=${token}`,
        method: "GET",
      }),
    }),
    registerEnterprise: builder.mutation<
      EnterpriseRegisterResponse,
      EnterpriseRegisterRequest
    >({
      query: ({
        email,
        password,
        agreedToTerms,
        userType,
        billingContact,
        company,
      }) => ({
        url: "auth/register",
        method: "POST",
        body: {
          email,
          password,
          agreedToTerms,
          userType,
          billingContact,
          company,
        },
      }),
    }),
    getUser: builder.query<GetUserResponse, void>({
      query: () => ({
        url: "user/",
        method: "GET",
      }),
    }),
    updateUser: builder.mutation<UpdateUserResponse, UpdateUserRequest>({
      query: (userData) => ({
        url: "user/update",
        method: "PUT",
        body: userData,
      }),
    }),
    deleteUser: builder.mutation<DeleteUserResponse, void>({
      query: () => ({
        url: "user/delete",
        method: "DELETE",
      }),
    }),
  }),
});

export const {
  useRegisterMutation,
  useLoginMutation,
  useForgotPasswordMutation,
  useResetPasswordMutation,
  useGoogleLoginMutation,
  useVerifyEmailQuery,
  useRegisterEnterpriseMutation,
  useGetUserQuery,
  useUpdateUserMutation,
  useDeleteUserMutation,
} = apiService;
