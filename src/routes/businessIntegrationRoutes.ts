import express from 'express';

import * as AdminController from '../controllers/admin.controller.js';
import { submitBusinessIntegration } from '../controllers/businessIntegration.controller.js';
import { businessIntegrationSchema } from '../lib/schemas/businessIntegration.schema.js';
import { authorizeRoles, protect } from '../middlewares/auth.js';
import { validateInput } from '../middlewares/validate.js';

const router = express.Router();

router.post(
  '/',
  validateInput(businessIntegrationSchema),
  submitBusinessIntegration
);

router.get(
  '/',
  protect,
  authorizeRoles('admin'),
  AdminController.getAllBusinessIntegrations
);

router.get(
  '/:id',
  protect,
  authorizeRoles('admin'),
  AdminController.getBusinessIntegrationById
);

router.patch(
  '/:id/status',
  protect,
  authorizeRoles('admin'),
  AdminController.updateBusinessIntegrationStatus
);

export default router;
