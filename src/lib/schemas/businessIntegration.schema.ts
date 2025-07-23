import { z } from 'zod';

const requiredString = (fieldName: string, min: number, max: number) =>
  z
    .string({
      required_error: `${fieldName} is required`,
    })
    .trim()
    .min(min, `${fieldName} must be at least ${min} characters`)
    .max(max, `${fieldName} cannot exceed ${max} characters`);

const emailSchema = z
  .string({ required_error: 'Email is required' })
  .trim()
  .toLowerCase()
  .email('Invalid email format')
  .regex(/^[^\s@]+@[^\s@]+\.[^\s@]+$/, 'Invalid email format');

const urlSchema = z
  .string({ required_error: 'Company website is required' })
  .trim()
  .regex(
    /^(https?:\/\/)?([\w.-]+)+(:\d+)?(\/[\w.-]*)*\/?$/,
    'Invalid website URL'
  );

const statusOptions = ['new', 'contacted', 'closed'] as const;

export const businessIntegrationSchema = z.object({
  firstName: requiredString('First name', 2, 50),
  lastName: requiredString('Last name', 2, 50),
  email: emailSchema,
  companyName: requiredString('Company name', 2, 100),
  companyWebsite: urlSchema,
  businessNeeds: requiredString('Business needs', 1, 500),
  status: z.enum(statusOptions).default('new'),
});

export type BusinessIntegrationType = z.infer<typeof businessIntegrationSchema>;
