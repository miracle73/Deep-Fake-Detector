import express from 'express';
const router = express.Router();
import * as BillingController from '../controllers/billing.controller.js';
import { protect } from '../middlewares/auth.js';
import { validateInput } from '../middlewares/validate.js';
import { checkoutSchema } from '../lib/schemas/billing.schema.js';

router.post(
  '/checkout',
  protect,
  validateInput(checkoutSchema),
  BillingController.createCheckoutSession
);

router.post('/portal', protect, BillingController.createCustomerPortal);

export default router;
