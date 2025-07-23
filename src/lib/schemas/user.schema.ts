import { z } from 'zod';

const baseUserSchema = z.object({
  email: z.string().email({ message: 'Invalid email address' }),
  password: z
    .string()
    .min(8, { message: 'Password must be at least 8 characters long' }),
  agreedToTerms: z.boolean().refine((val) => val === true, {
    message: 'You must agree to the terms and conditions',
  }),
  plan: z
    .enum(['SafeGuard_Free', 'SafeGuard_Pro', 'SafeGuard_Max'])
    .optional()
    .default('SafeGuard_Free'),
});

export const individualUserSchema = baseUserSchema.extend({
  userType: z.literal('individual'),
  firstName: z.string().min(1, { message: 'First name is required' }),
  lastName: z.string().min(1, { message: 'Last name is required' }),
});

const companySchema = z.object({
  name: z.string().min(1, { message: 'Company name is required' }),
  website: z.string().url({ message: 'Valid website URL is required' }),
  size: z.string().optional(),
  industry: z.string().optional(),
});

const billingContactSchema = z.object({
  name: z.string().min(1, { message: 'Contact name is required' }),
  email: z.string().email({ message: 'Valid contact email is required' }),
  phone: z.string().min(1, { message: 'Contact phone is required' }),
});

export const enterpriseUserSchema = baseUserSchema.extend({
  userType: z.literal('enterprise'),
  company: companySchema,
  billingContact: billingContactSchema,
});

export const registerSchema = z.discriminatedUnion('userType', [
  individualUserSchema,
  enterpriseUserSchema,
]);

export const loginSchema = z.object({
  email: z.string().email({ message: 'Invalid email address' }),
  password: z.string().min(1, { message: 'Password is required' }),
});

export const forgotPasswordSchema = z.object({
  email: z.string().email({ message: 'Invalid email address' }),
});

export const resetPasswordSchema = z.object({
  resetToken: z.string().min(1, { message: 'Reset token is required' }),
  newPassword: z
    .string()
    .min(8, { message: 'New password must be at least 8 characters long' }),
  confirmPassword: z
    .string()
    .min(8, { message: 'Confirm password must be at least 8 characters long' }),
});

export const resetPasswordBodySchema = z.object({
  password: z
    .string()
    .min(8, { message: 'Password must be at least 8 characters long' }),
});

export const resendVerificationEmailSchema = z.object({
  email: z.string().email({ message: 'Valid email address is required' }),
});

export const verifyEmailQuerySchema = z.object({
  token: z.string().min(1, { message: 'Verification token is required' }),
});

export const updateUserSchema = z.object({
  lastName: z.string().min(2).optional(),
  firstName: z.string().min(2).optional(),
  email: z.string().email().optional(),
  currentPassword: z.string().optional(),
  newPassword: z.string().min(8).optional(),
});

export const updateSubscriptionSchema = z.object({
  emailSubscribed: z.boolean(),
});

export type RegisterInput = z.infer<typeof registerSchema>;
export type LoginInput = z.infer<typeof loginSchema>;
export type ForgotPasswordInput = z.infer<typeof forgotPasswordSchema>;
export type UpdateInput = z.infer<typeof updateUserSchema>;
export type ResetPasswordBodyInput = z.infer<typeof resetPasswordBodySchema>;
export type ResendVerificationEmailInput = z.infer<
  typeof resendVerificationEmailSchema
>;
export type VerifyEmailQueryInput = z.infer<typeof verifyEmailQuerySchema>;
