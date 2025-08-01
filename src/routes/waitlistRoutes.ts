import { Router } from 'express';
import * as WaitlistController from '../controllers/waitlist.controller.js';
import {
  waitlistSignupLimiter,
  waitlistStatusLimiter,
} from '../middlewares/rateLimit.js';
import {
  waitlistSignupSchema,
  waitlistStatusSchema,
} from '../lib/schemas/waitlist.schema.js';
import { validate, validateInput } from '../middlewares/validate.js';
import { authorize, authorizeRoles, protect } from '../middlewares/auth.js';

const router = Router();

router.post(
  '/signup',
  waitlistSignupLimiter,
  validateInput(waitlistSignupSchema),
  WaitlistController.SignupForWaitlist
);

router.get(
  '/status',
  waitlistStatusLimiter,
  validate(waitlistStatusSchema),
  WaitlistController.getWaitlistStatus
);

router.get(
  '/stats',
  protect,
  authorizeRoles('admin'),
  WaitlistController.getWaitlistStats
);

export default router;
