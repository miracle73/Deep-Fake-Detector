import Stripe from 'stripe';
import { config } from 'dotenv';
import User from '../models/User.js';
import Subscription from '../models/Subscription.js';
import type { IUser } from '../types/user.js';
import PaymentHistory from '../models/PaymentHistory.js';

config();

const stripeApiKey =
  process.env.STRIPE_SECRET_KEY ||
  'sk_test_placeholder sympathiqueBuildProcess';

export const stripe = new Stripe(stripeApiKey as string, {
  apiVersion: '2025-04-30.basil',
});

export async function handleSubscriptionUpdate(
  sub: Stripe.Subscription,
  user: IUser
) {
  const price = sub.items.data[0].price;

  const subscription = await Subscription.findOneAndUpdate(
    { stripeSubscriptionId: sub.id },
    {
      user: user._id,
      stripeSubscriptionId: sub.id,
      status: sub.status,
      priceId: price.id,
      productId: price.product as string,
      plan: price.nickname?.toLowerCase() ?? 'unknown',
      startDate: new Date(sub.start_date * 1000),
      currentPeriodStart: new Date(sub.start_date * 1000),
      currentPeriodEnd: sub.cancel_at ? new Date(sub.cancel_at * 1000) : null,
      cancelAtPeriodEnd: sub.cancel_at_period_end,
    },
    { upsert: true, new: true }
  );

  return subscription;
}

export async function handleSuccessfulPayment(invoice: Stripe.Invoice) {
  // TODO: Implement payment handling
  // const customer = await stripe.customers.retrieve(invoice.customer as string);
  // const userId = customer.metadata.userId;
  // await PaymentHistory.create({
  //   userId,
  //   stripePaymentIntentId: invoice.payment_intent as string,
  //   amount: invoice.amount_paid,
  //   currency: invoice.currency,
  //   status: 'succeeded',
  //   paymentMethod: invoice.payment_method_types?.[0] || 'card',
  // });
}

export async function handleFailedPayment(invoice: Stripe.Invoice) {
  // const customer = await stripe.customers.retrieve(invoice.customer as string);
  // const userId = customer.metadata.userId;
  // await PaymentHistory.create({
  //   userId,
  //   stripePaymentIntentId: invoice.payment_intent as string,
  //   amount: invoice.amount_due,
  //   currency: invoice.currency,
  //   status: 'failed',
  //   paymentMethod: invoice.payment_method_types?.[0] || 'card',
  // });
  // const subscription = await Subscription.findOne({
  //   userId,
  //   status: 'active',
  // });
  // if (subscription) {
  //   subscription.status = 'past_due';
  //   await subscription.save();
  // }
}
