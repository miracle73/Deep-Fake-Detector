import { z } from 'zod';

export const waitlistSignupSchema = z.object({
  email: z
    .string()
    .email('Please provide a valid email address')
    .min(5, 'Email must be at least 5 characters')
    .max(254, 'Email must not exceed 254 characters')
    .transform((email) => email.toLowerCase().trim()),
});

export const waitlistStatusSchema = z.object({
  query: z.object({
    email: z
      .string()
      .email('Please provide a valid email address')
      .transform((email) => email.toLowerCase().trim()),
  }),
});

export type WaitlistSignup = z.infer<typeof waitlistSignupSchema>;
export type WaitlistStatus = z.infer<typeof waitlistSignupSchema>;
