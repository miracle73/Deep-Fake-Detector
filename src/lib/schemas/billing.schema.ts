import { z } from 'zod';

export const checkoutSchema = z.object({
  priceId: z.string().min(1, { message: 'Price ID is required' }),
});

export type checkoutSchema = z.infer<typeof checkoutSchema>;
