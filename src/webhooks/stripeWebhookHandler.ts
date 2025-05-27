import { stripe } from '../services/stripeService';
import type { Request, Response } from 'express';
import type Stripe from 'stripe';

export const handleStripeWebhook = async (
  req: Request,
  res: Response
): Promise<any> => {
  const sig = req.headers['stripe-signature'] as string;
  const endpointSecret = process.env.STRIPE_WEBHOOK_SECRET as string;

  if (!sig || !endpointSecret) {
    return res.status(400).send('Missing Stripe signature or endpoint secret');
  }

  let event: Stripe.Event;

  try {
    event = stripe.webhooks.constructEvent(req.body, sig, endpointSecret);
  } catch (error) {
    return res
      .status(400)
      .send(`Webhook Error: ${(error as Error)?.message || 'Unknown error'}`);
  }

  switch (event.type) {
    case 'checkout.session.completed': {
      const session = event.data.object;
      // Lookup user by session.customer and update their plan
      break;
    }

    case 'invoice.payment_failed':
      // Downgrade or notify user
      break;

    case 'customer.subscription.deleted':
      // Mark user plan as "free"
      break;

    default:
      break;
  }
  res.status(200).json({
    success: true,
    code: 200,
    message: 'Webhook handled successfully',
    received: true,
  });
};
