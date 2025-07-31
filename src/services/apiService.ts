import { createApi, fetchBaseQuery } from "@reduxjs/toolkit/query/react";

interface RegisterRequest {
  email: string;
  password: string;
  firstName: string;
  lastName: string;
  agreedToTerms: boolean;
  userType: string;
  // phoneNumber: string;
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

export interface AnalysisHistory {
  id: string;
  type: string;
  result: string;
  createdAt: string;
  fileName?: string;
  confidence?: number;
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

interface SubscriptionPlan {
  id: string;
  object: string;
  active: boolean;
  attributes: unknown[];
  created: number;
  default_price: string;
  description: string;
  images: unknown[];
  livemode: boolean;
  marketing_features: unknown[];
  metadata: Record<string, unknown>;
  name: string;
  package_dimensions: unknown;
  shippable: unknown;
  statement_descriptor: unknown;
  tax_code: unknown;
  type: string;
  unit_label: unknown;
  updated: number;
  url: unknown;
}

interface SubscriptionPlansResponse {
  success: boolean;
  message: string;
  data: {
    object: string;
    data: SubscriptionPlan[];
    has_more: boolean;
    url: string;
  };
}

interface CheckoutRequest {
  priceId: string;
}

interface CheckoutResponse {
  success: boolean;
  code: number;
  message: string;
  data: {
    sessionId: string;
    sessionUrl: string;
    amount: number;
  };
}

interface UpdateSubscriptionRequest {
  emailSubscribed: boolean;
}

interface UpdateSubscriptionResponse {
  success: boolean;
  message: string;
  data: {
    newStatus: boolean;
  };
}

interface MarkNotificationReadResponse {
  success: boolean;
  message?: string;
}

interface UpdateMediaConsentRequest {
  allowStorage: boolean;
}

interface UpdateMediaConsentResponse {
  success: boolean;
  message: string;
  data: {
    consent: {
      storeMedia: boolean;
      updatedAt: string;
    };
  };
}

interface Notification {
  _id: string;
  userId: string;
  type: string;
  title: string;
  message: string;
  read: boolean;
  expiresAt: string;
  createdAt: string;
  updatedAt: string;
  __v: number;
  id: string;
}

interface GetNotificationsResponse {
  success: boolean;
  message: string;
  notifications: Notification[];
}

interface DetectAnalyzeRequest {
  image: File;
}

export interface DetectAnalyzeResponse {
  statusCode: number;
  status: string;
  success: boolean;
  message: string;
  data: {
    confidence: number;
    deepfake_probability: number;
    is_deepfake: boolean;
    predicted_class: string;
    real_probability: number;
    threshold_used: number;
  };
}

interface DemoRequestResponse {
  success: boolean;
  user: {
    id: string;
    email: string;
    userType: string;
    plan: string;
    phoneNumber: string;
    firstName: string;
    lastName: string;
  };
}

interface AnalysisHistoryItem {
  _id: string;
  userId: string;
  fileName: string;
  thumbnailUrl: string;
  uploadDate: string;
  status: string;
  confidenceScore: number;
  __v: number;
}

interface GetAnalysisHistoryResponse {
  success: boolean;
  analysisHistory: AnalysisHistoryItem[];
}
export const apiService = createApi({
  reducerPath: "apiService",
  baseQuery: fetchBaseQuery({
    baseUrl:
      "https://deepfake-detector-996819843496.us-central1.run.app/api/v1",

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
        body: {
          email,
          password,
          firstName,
          lastName,
          agreedToTerms,
          userType,
        },
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
    subscriptionPlans: builder.query<SubscriptionPlansResponse, void>({
      query: () => ({
        url: "subscriptions/plans",
        method: "GET",
      }),
    }),
    checkout: builder.mutation<CheckoutResponse, CheckoutRequest>({
      query: ({ priceId }) => ({
        url: "subscriptions/checkout",
        method: "POST",
        body: { priceId },
      }),
    }),
    updateEmailSubscription: builder.mutation<
      UpdateSubscriptionResponse,
      UpdateSubscriptionRequest
    >({
      query: ({ emailSubscribed }) => ({
        url: "user/subscription",
        method: "PATCH",
        body: { emailSubscribed },
      }),
    }),
    markNotificationRead: builder.mutation<
      MarkNotificationReadResponse,
      string
    >({
      query: (notificationId) => ({
        url: `notifications/${notificationId}/read`,
        method: "PATCH",
      }),
    }),
    updateMediaConsent: builder.mutation<
      UpdateMediaConsentResponse,
      UpdateMediaConsentRequest
    >({
      query: ({ allowStorage }) => ({
        url: "user/consent",
        method: "PATCH",
        body: { allowStorage },
      }),
    }),
    getNotifications: builder.query<GetNotificationsResponse, void>({
      query: () => ({
        url: "notifications",
        method: "GET",
      }),
    }),
    detectAnalyze: builder.mutation<
      DetectAnalyzeResponse,
      DetectAnalyzeRequest
    >({
      query: ({ image }) => {
        const formData = new FormData();
        formData.append("image", image);

        return {
          url: "detect/analyze",
          method: "POST",
          body: formData,
        };
      },
    }),
    createPassword: builder.mutation<
      DemoRequestResponse,
      { token: string; password: string }
    >({
      query: ({ token, password }) => ({
        url: `demo-request/user?token=${token}`,
        method: "POST",
        body: { password },
      }),
    }),
    getAnalysisHistory: builder.query<GetAnalysisHistoryResponse, void>({
      query: () => ({
        url: "user/analysis-history",
        method: "GET",
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
  useSubscriptionPlansQuery,
  useCheckoutMutation,
  useUpdateEmailSubscriptionMutation,
  useMarkNotificationReadMutation,
  useUpdateMediaConsentMutation,
  useGetNotificationsQuery,
  useDetectAnalyzeMutation,
  useCreatePasswordMutation,
  useGetAnalysisHistoryQuery,
} = apiService;
