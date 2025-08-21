import { Router } from 'express';

import { FeedbackController } from '../controllers/feedback.controller.js';
import {
  createFeedbackSchema,
  updateFeedbackSchema,
} from '../lib/schemas/feedback.schema.js';
import { authorizeRoles, protect } from '../middlewares/auth.js';
import { validateInput } from '../middlewares/validate.js';

const router = Router();

const feedbackController = new FeedbackController();

// Public routes
router.post(
  '/',
  validateInput(createFeedbackSchema),
  feedbackController.createFeedback
);

router.use(protect);

// Admin routes
router.get('/', authorizeRoles('admin'), feedbackController.getAllFeedback);
router.get(
  '/stats',
  authorizeRoles('admin'),
  feedbackController.getFeedbackStats
);
router.get('/:id', authorizeRoles('admin'), feedbackController.getFeedback);
router.put(
  '/:id',
  authorizeRoles('admin'),
  validateInput(updateFeedbackSchema),
  feedbackController.updateFeedback
);
router.delete(
  '/:id',
  authorizeRoles('admin'),
  feedbackController.deleteFeedback
);

export { router as feedbackRoutes };
