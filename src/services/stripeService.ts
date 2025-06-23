import Stripe from 'stripe';
import { config } from 'dotenv';
import User from '../models/User.js';
import Subscription from '../models/Subscription.js';
import type { IUser } from '../types/user.js';
import PaymentHistory from '../models/PaymentHistory.js';
import logger from '../utils/logger.js';
import { sendEmail } from './emailService.js';

config();

const PLAN_QUOTA = {
  Free: 3,
  Pro: 30,
  Max: Number.POSITIVE_INFINITY,
};

const stripeApiKey =
  process.env.STRIPE_SECRET_KEY ||
  'sk_test_placeholder sympathiqueBuildProcess';

export const stripe = new Stripe(stripeApiKey as string, {
  apiVersion: '2025-04-30.basil',
});

export const handleCheckoutSessionCompleted = async (
  session: Stripe.Checkout.Session
) => {
  try {
    if (session.mode !== 'subscription') return;

    if (!session.customer || typeof session.customer !== 'string') {
      throw new Error('Missing customer ID in checkout session');
    }

    const subscription = await stripe.subscriptions.retrieve(
      session.subscription as string
    );

    const price = subscription.items.data[0].price;
    const productId = price.product as string;

    const product = await stripe.products.retrieve(productId);
    const productName = product.name;

    const user = await User.findOneAndUpdate(
      { stripeCustomerId: session.customer },
      {
        plan: productName || 'SAFEGUARD_PRO',
        isActive: true,
        // $push: {
        //   billingHistory: {
        //     invoiceId: session.invoice as string,
        //     amount: session.amount_total ? session.amount_total / 100 : 0,
        //     plan: subscription.items.data[0].price.lookup_key || 'pro',
        //     status: 'paid',
        //     date: new Date(),
        //   },
        // },
      },
      { new: true }
    );

    if (!user) {
      throw new Error(`User not found for customer ID: ${session.customer}`);
    }

    await Subscription.findOneAndUpdate(
      { stripeSubscriptionId: subscription.id },
      {
        userId: user._id,
        status: subscription.status,
        planId: subscription.items.data[0].price.id,
        currentPeriodStart: new Date(subscription.start_date * 1000),
        currentPeriodEnd: subscription.cancel_at
          ? new Date(subscription.cancel_at * 1000)
          : null,
        cancelAtPeriodEnd: subscription.cancel_at_period_end,
      },
      { upsert: true, new: true }
    );

    logger.info(`Checkout completed for user ${user.email}`);
  } catch (error) {
    logger.error('Failed to handle checkout.session.completed', error);
    throw error;
  }
};

export const handleSubscriptionChange = async (
  subscription: Stripe.Subscription
) => {
  const user = await User.findOne({
    stripeCustomerId: subscription.customer as string,
  });
  if (!user) return;

  // Update user plan status
  const plan = subscription.items.data[0].price.lookup_key || user.plan;

  await User.updateOne(
    { _id: user._id },
    {
      plan,
      isActive: ['active', 'trialing'].includes(subscription.status),
    }
  );

  await Subscription.findOneAndUpdate(
    { stripeSubscriptionId: subscription.id },
    {
      status: subscription.status,
      planId: subscription.items.data[0].price.id,
      currentPeriodStart: new Date(subscription.start_date * 1000),
      currentPeriodEnd: subscription.cancel_at
        ? new Date(subscription.cancel_at * 1000)
        : null,
      cancelAtPeriodEnd: subscription.cancel_at_period_end,
    }
  );

  logger.info(
    `Subscription ${subscription.id} updated to status: ${subscription.status}`
  );
};

export const handleSubscriptionCancelled = async (
  subscription: Stripe.Subscription
) => {
  await Subscription.findOneAndUpdate(
    { stripeSubscriptionId: subscription.id },
    {
      status: 'canceled',
      canceledAt: new Date(),
      cancelAtPeriodEnd: false,
    }
  );

  // Optionally downgrade user immediately or wait until period ends
  const user = await User.findOne({
    stripeCustomerId: subscription.customer as string,
  });
  if (user) {
    await User.updateOne(
      { _id: user._id },
      {
        plan: 'free',
        isActive: false,
        $push: {
          billingHistory: {
            invoiceId: `cancel_${Date.now()}`,
            status: 'canceled',
            date: new Date(),
          },
        },
      }
    );
  }

  logger.info(`Subscription ${subscription.id} canceled`);
};

export const handleSuccessfulPayment = async (invoice: Stripe.Invoice) => {
  const subscriptions = await stripe.subscriptions.list({
    limit: 3,
  });
  const subscription = await stripe.subscriptions.retrieve(
    invoice.parent?.subscription_details?.subscription as string
  );

  const price = subscription.items.data[0].price;
  const productId = price.product as string;

  const product = await stripe.products.retrieve(productId);

  const productName = product.name;

  const user = await User.findOne({
    stripeCustomerId: invoice.customer as string,
  });

  if (!user) return;

  await User.updateOne(
    { _id: user._id },
    {
      $set: {
        usageQuota: {
          monthlyAnalysis: 30,
          remainingAnalysis: user.usageQuota.remainingAnalysis + 30,
          lastResetAt: new Date(),
          carryOver: true,
        },
      },
      $push: {
        billingHistory: {
          invoiceId: invoice.id,
          amount: invoice.amount_paid / 100,
          plan: productName,
          status: 'paid',
          date: new Date(),
        },
      },
    }
  );

  // await sendEmail({
  //   to: user.email,
  //   subject: 'Payment Receipt',
  //   text: `payment-receipt - amount:  ${
  //     invoice.amount_paid / 100
  //   } date - ${new Date(
  //     invoice.created * 1000
  //   ).toLocaleDateString()} invoiceUrl- ${invoice.hosted_invoice_url}`,
  // });

  logger.info(`Payment succeeded for invoice ${invoice.id}`);
};

export const handleFailedPayment = async (invoice: Stripe.Invoice) => {
  const user = await User.findOne({
    stripeCustomerId: invoice.customer as string,
  });
  if (!user) return;

  await User.updateOne(
    { _id: user._id },
    {
      $push: {
        billingHistory: {
          invoiceId: invoice.id,
          amount: invoice.amount_due / 100,
          plan: user.plan,
          status: 'failed',
          date: new Date(),
        },
      },
    }
  );

  // Optionally send payment failure email
  logger.warn(`Payment failed for invoice ${invoice.id}`);
};

export const handleUpcomingInvoice = async (invoice: Stripe.Invoice) => {
  const user = await User.findOne({
    stripeCustomerId: invoice.customer as string,
  });
  if (!user) return;

  const subscription = await stripe.subscriptions.retrieve(
    invoice.parent?.subscription_details?.subscription as string
  );

  const plan = subscription.items.data[0].price.lookup_key || user.plan;
  const amount = invoice.amount_due / 100;
  const chargeDate = invoice.next_payment_attempt
    ? new Date(invoice.next_payment_attempt * 1000)
    : null;

  // Add to queue
  // await queueEmailNotification.add('upcoming-invoice-email', {
  //   to: user.email,
  //   subject: 'Upcoming Subscription Payment',
  //   template: 'upcoming-invoice',
  //   data: {
  //     amount,
  //     plan,
  //     chargeDate: chargeDate.toDateString(),
  //   },
  // });

  logger.info(`Queued upcoming invoice email for ${user.email}`);
};
