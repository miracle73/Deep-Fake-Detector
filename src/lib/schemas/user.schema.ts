import { z } from 'zod';

export const registerSchema = z.object({
  email: z.string().email({ message: 'Invalid email address' }),
  password: z
    .string()
    .min(8, { message: 'Password must be at least 8 characters long' }),
  firstName: z.string().min(1, { message: 'First name is required' }),
  lastName: z.string().min(1, { message: 'Last name is required' }),
  agreedToTerms: z.boolean().refine((val) => val === true, {
    message: 'You must agree to the terms and conditions',
  }),
  userType: z.enum(['individual', 'enterprise']),
  plan: z.string().optional().default('free'),
});

export const loginSchema = z.object({
  email: z.string().email({ message: 'Invalid email address' }),
  password: z
    .string()
    .min(8, { message: 'Password must be at least 8 characters long' }),
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

export type RegisterInput = z.infer<typeof registerSchema>;
export type LoginInput = z.infer<typeof loginSchema>;
export type ForgotPasswordInput = z.infer<typeof forgotPasswordSchema>;
