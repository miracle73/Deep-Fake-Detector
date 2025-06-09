import Stripe from 'stripe';
import { config } from 'dotenv';
import User from '../models/User.js';
import Subscription from '../models/Subscription.js';

config();

const stripeApiKey =
  process.env.STRIPE_SECRET_KEY ||
  'sk_test_placeholder sympathiqueBuildProcess';

export const stripe = new Stripe(stripeApiKey as string, {
  apiVersion: '2025-04-30.basil',
});

export async function handleSubscriptionUpdate(
  stripeSubscription: Stripe.Subscription
) {
  const customer = await stripe.customers.retrieve(
    stripeSubscription.customer as string
  );

  let userId;
  if ('metadata' in customer && customer.metadata?.userId) {
    userId = customer.metadata.userId;
    // TODO: Handle the userId
  }

  // const subscription = await Subscription.findOneAndUpdate(
  //   { stripeSubscriptionId: stripeSubscription.id },
  //   {
  //     userId,
  //     status: stripeSubscription.status,
  //     currentPeriodStart: new Date(
  //       stripeSubscription.current_period_start * 1000
  //     ),
  //     currentPeriodEnd: new Date(stripeSubscription.current_period_end * 1000),
  //     cancelAtPeriodEnd: stripeSubscription.cancel_at_period_end,
  //   },
  //   { upsert: true, new: true }
  // );

  // await User.findByIdAndUpdate(userId, {
  //   currentSubscriptionPlan: subscription.planId,
  // });
}

export async function handleSuccessfulPayment(invoice: Stripe.Invoice) {
  // TODO: Implement payment handling
}

export async function handleFailedPayment(invoice: Stripe.Invoice) {}
