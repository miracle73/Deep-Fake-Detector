export const subscriptionPlans = {
  free: {
    id: 'free',
    name: 'Safeguard Free',
    price: 0,
    features: [
      'Access to web interface only',
      '50 media analyses per month',
      'Community support',
      'Limited model accuracy',
    ],
    stripeProductId: null,
    stripePriceId: null,
    requestLimit: 50,
  },

  pro: {
    id: 'pro',
    name: 'Safeguard Pro',
    price: 29,
    features: [
      'API + web access',
      '5,000 media analyses/month',
      'High-accuracy detection model',
      'Detection confidence scores',
      'Email support',
    ],
    stripeProductId: 'prod_safeguard_pro',
    stripePriceId: 'price_safeguard_pro',
    requestLimit: 500,
  },

  max: {
    id: 'max',
    name: 'Safeguard Max',
    price: 89,
    features: [
      'Unlimited access to all features',
      'Unlimited media analyses',
      'Priority API throughput',
      'Advanced analytics dashboard',
      'Batch processing support',
      'Premium email & chat support',
    ],
    stripeProductId: 'prod_safeguard_max',
    stripePriceId: 'price_safeguard_max',
    requestLimit: Infinity,
  },
};
