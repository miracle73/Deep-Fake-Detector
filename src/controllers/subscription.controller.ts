import Subscription from '../models/Subscription.js';
import {
  handleCheckoutSessionCompleted,
  handleFailedPayment,
  handleSubscriptionCancelled,
  handleSubscriptionChange,
  handleSuccessfulPayment,
  handleUpcomingInvoice,
  stripe,
} from '../services/stripeService.js';
import { AppError, NotFoundError } from '../utils/error.js';
import logger from '../utils/logger.js';

import type Stripe from 'stripe';

import type { AuthRequest } from '../middlewares/auth.js';
import type { checkoutSchema } from '../lib/schemas/billing.schema.js';
import type { NextFunction, Request, Response } from 'express';
import WebhookEvent from '../models/WebhookEvent.js';

export const getSubscriptionsPlan = async (
  req: Request,
  res: Response,
  next: NextFunction
) => {
  try {
    const products = await stripe.products.list({});

    res.status(200).json({
      success: true,
      message: 'Subscription plans fetched successfully',
      data: products,
    });
  } catch (error) {
    logger.error('Failed to fetch subscription plans', error);
    next(error);
  }
};

export const createCheckoutSession = async (
  req: AuthRequest,
  res: Response,
  next: NextFunction
) => {
  try {
    const user = req.user;
    const { priceId } = req.body as checkoutSchema;

    if (!priceId) {
      throw new AppError(400, 'Price ID is required in the request body', null);
    }

    const allowedPriceIds = [
      process.env.STRIPE_PRICE_FREE,
      process.env.STRIPE_PRICE_PRO,
      process.env.STRIPE_PRICE_MAX,
    ];

    if (!allowedPriceIds.includes(priceId)) {
      throw new AppError(400, 'Invalid priceId passed');
    }

    if (!user) {
      throw new NotFoundError('User not found');
    }

    let stripeCustomerId = user.stripeCustomerId;

    if (!stripeCustomerId) {
      const customer = await stripe.customers.create({
        email: user.email,
        metadata: {
          userId: user._id.toString(),
          appUserType: user.userType,
        },
      });
      stripeCustomerId = customer.id;
      user.stripeCustomerId = stripeCustomerId;
      await user.save();
    }

    const session = await stripe.checkout.sessions.create({
      customer: user.stripeCustomerId,
      // invoice_creation: true,
      payment_method_types: ['card'],
      mode: 'subscription',
      metadata: {
        userId: user._id.toString(),
        email: user.email,
        planIntent: priceId,
      },
      line_items: [{ price: priceId, quantity: 1 }],
      success_url: `${process.env.FRONTEND_URL}/billing/success/`,
      cancel_url: `${process.env.FRONTEND_URL}/billing/cancelled`,
    });

    if (!session) {
      throw new AppError(400, 'Failed to create checkout session', session);
    }

    res.status(200).json({
      success: true,
      code: 200,
      message: 'Checkout session created successfully',
      data: {
        sessionId: session.id,
        sessionUrl: session.url,
        amount: session.amount_total ? session.amount_total / 100 : null,
      },
    });
  } catch (error) {
    logger.error('Error creating checkout session:', error);
    next(error);
  }
};

export const cancelSubscription = async (
  req: AuthRequest,
  res: Response,
  next: NextFunction
) => {
  try {
    const userId = req.user._id;

    const subscription = await Subscription.findOne({
      userId,
      status: 'active',
    });

    if (!subscription) {
      throw new NotFoundError('No active subscription found');
    }

    await stripe.subscriptions.update(subscription.stripeSubscriptionId, {
      cancel_at_period_end: true,
    });

    subscription.cancelAtPeriodEnd = true;
    await subscription.save();

    res.status(201).json({
      success: true,
      code: 201,
      message: 'Subscription cancelled successfully',
    });
  } catch (error) {
    logger.error('Failed to cancel subscription', error);
    next(error);
  }
};

export const getCurrentSubscription = async (
  req: AuthRequest,
  res: Response,
  next: NextFunction
) => {
  try {
    const userId = req.user._id;

    const subscription = await Subscription.findOne({ userId }).sort({
      createdAt: -1,
    });

    if (!subscription) {
      res.status(201).json({
        success: true,
        message: 'Current subscription fetched successfully',
        data: {
          plan: 'SafeGuard_Free',
        },
      });
    }

    res.status(200).json({
      success: true,
      message: 'Current subscription fetched successfully',
      data: {
        plan: subscription,
      },
    });
  } catch (error) {
    logger.error('Failed to fetch current subscription', error);
    next(error);
  }
};

export const handleStripeWebhook = async (
  req: AuthRequest,
  res: Response,
  next: NextFunction
) => {
  try {
    const sig = req.headers['stripe-signature'] as string;
    const rawBody = req.body;

    let event: Stripe.Event;

    try {
      event = stripe.webhooks.constructEvent(
        rawBody,
        sig,
        process.env.STRIPE_WEBHOOK_SECRET || ''
      );

      const existingEvent = await WebhookEvent.findOne({ eventId: event.id });
      if (existingEvent) {
        logger.warn(`Duplicate webhook event ${event.id}`);
        return; // { processed: false, reason: 'duplicate' };
      }
    } catch (error) {
      logger.error(
        'Webhook Error',
        error instanceof Error ? error.message : 'Unknown error'
      );
      throw new AppError(
        400,
        'Webhook Error',
        error instanceof Error ? error.message : 'Unknown error'
      );
    }

    switch (event.type) {
      case 'checkout.session.completed':
        await handleCheckoutSessionCompleted(
          event.data.object as Stripe.Checkout.Session
        );

        // send email
        break;

      case 'customer.subscription.created':
      case 'customer.subscription.updated':
        await handleSubscriptionChange(
          event.data.object as Stripe.Subscription
        );
        break;

      case 'customer.subscription.deleted':
        await handleSubscriptionCancelled(
          event.data.object as Stripe.Subscription
        );
        break;

      case 'invoice.upcoming':
        await handleUpcomingInvoice(event.data.object as Stripe.Invoice);
        break;

      case 'invoice.payment_succeeded':
        await handleSuccessfulPayment(
          event.data.object as Stripe.Invoice,
          event
        );
        break;

      case 'invoice.payment_failed':
        await handleFailedPayment(event.data.object as Stripe.Invoice);
        break;

      default:
        logger.info(`Unhandled event type: ${event.type}`);
    }

    res.status(200).json({
      success: true,
      message: 'Stripe webhook received and processed successfully',
      data: {
        received: true,
      },
    });
  } catch (error) {
    logger.error('Failed to handle stripe webhook', error);
    next(error);
  }
};

export const createCustomerPortal = async (
  req: Request,
  res: Response,
  next: NextFunction
) => {
  try {
    const user = req.user;

    if (!user || !user.stripeCustomerId) {
      throw new AppError(
        401,
        'User is not authenticated or does not have a Stripe customer ID'
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
    logger.error('Error creating customer portal:', error);
    next(error);
  }
};
