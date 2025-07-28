import { z } from 'zod';

// Define reusable string validations
const requiredString = (fieldName: string) =>
  z
    .string({
      required_error: `${fieldName} is required`,
    })
    .trim()
    .min(3, `${fieldName} must be at least 3 characters`)
    .max(50, `${fieldName} cannot exceed 50 characters`);

const emailSchema = z
  .string({
    required_error: 'Email is required',
  })
  .email('Invalid email format')
  .toLowerCase()
  .trim();

// Define enum options (matching Mongoose enums)
const roleOptions = [
  'Content Creator/Influencer',
  'Journalist/Reporter',
  'Educator/Teacher',
  'Researcher/Academic',
  'Freelancer/Consultant',
  'Student',
  'Individual User',
  'Other',
] as const;

const goalOptions = [
  'Verify content i receive',
  'Protect my personal brand',
  'Fact-checking and research',
  'Teaching/learning about deepfakes',
  'Detect manipulated media',
  'General digital security',
  'Just curious about the technology',
] as const;

const contentTypeOptions = [
  'Videos',
  'Images/Photos',
  'Audio content',
  'Social media posts',
  'News articles',
  'User-generated content',
  'Mixed content types',
] as const;

const urgencyLevelOptions = [
  'Need help right now',
  'Within this week',
  'Within this month',
  'Planning for the future',
  'Just exploring options',
] as const;

// Main schema
export const demoRequestSchema = z.object({
  firstName: requiredString('First name'),
  lastName: requiredString('Last name'),
  email: emailSchema,
  role: z.enum(roleOptions).default('Other'),
  goal: z.enum(goalOptions).default('Detect manipulated media'),
  contentType: z.enum(contentTypeOptions).optional(),
  urgencyLevel: z.enum(urgencyLevelOptions).optional(),
  metadata: z.string().default('').optional(),
});

export const demoUserSchema = z.object({
  password: z
    .string()
    .min(8, { message: 'Password must be at least 8 characters long' }),
  email: emailSchema,
});

export type DemoRequest = z.infer<typeof demoRequestSchema>;

export type DemoUser = z.infer<typeof demoUserSchema>;
