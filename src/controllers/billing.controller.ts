import { AppError } from '../middlewares/error.js';
import { stripe } from '../services/stripeService.js';

import type { checkoutSchema } from '../lib/schemas/billing.schema.js';
import type { Request, Response } from 'express';

export const createCheckoutSession = async (req: Request, res: Response) => {
  try {
    const user = req.user;
    const { priceId } = req.body as checkoutSchema;

    if (!user || !user.stripeCustomerId) {
      throw new AppError(
        'User is not authenticated or does not have a Stripe customer ID',
        401
      );
    }

    if (!priceId) {
      throw new AppError('Price ID is required', 400);
    }

    const session = await stripe.checkout.sessions.create({
      customer: user.stripeCustomerId,
      payment_method_types: ['card'],
      mode: 'subscription',
      line_items: [{ price: priceId, quantity: 1 }],
      success_url: `${process.env.FRONTEND_URL}/dashboard`,
      cancel_url: `${process.env.FRONTEND_URL}/billing/cancelled`,
    });

    res.status(200).json({
      success: true,
      code: 200,
      message: 'checkout session created successfully',
      sessionId: session.id,
    });
  } catch (error) {
    console.error('Error creating checkout session:', error);
    res.status(500).json({
      success: false,
      code: 500,
      mesage: 'Failed to create checkout session',
      error: error instanceof Error ? error.message : 'Unknown error',
    });
  }
};

export const createCustomerPortal = async (req: Request, res: Response) => {
  try {
    const user = req.user;

    if (!user || !user.stripeCustomerId) {
      throw new AppError(
        'User is not authenticated or does not have a Stripe customer ID',
        401
      );
    }

    const session = await stripe.billingPortal.sessions.create({
      customer: user.stripeCustomerId,
      return_url: `${process.env.FRONTEND_URL}/dashboard`,
    });

    res.json({
      success: true,
      message: 'Customer portal created successfully',
      url: session.url,
    });
  } catch (error) {
    console.error('Error creating customer portal:', error);
    res.status(500).json({
      success: false,
      code: 500,
      mesage: 'Failed to create customer portal',
      error: error instanceof Error ? error.message : 'Unknown error',
    });
  }
};
