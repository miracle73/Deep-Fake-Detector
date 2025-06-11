import express from 'express';

import * as SubscriptionController from '../controllers/subscription.controller.js';
import { checkoutSchema } from '../lib/schemas/billing.schema.js';
import { protect } from '../middlewares/auth.js';
import { validateInput } from '../middlewares/validate.js';

const router = express.Router();

router.get('/plans', protect, SubscriptionController.getSubscriptionsPlan);

router.post(
  '/checkout',
  protect,
  validateInput(checkoutSchema),
  SubscriptionController.createCheckoutSession
);

router.post('/portal', protect, SubscriptionController.createCustomerPortal);

router.post('/webhook', SubscriptionController.handleStripeWebhook);

export default router;
