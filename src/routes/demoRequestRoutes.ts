import express from 'express';

import * as AdminController from '../controllers/admin.controller.js';
import * as DemoRequestController from '../controllers/demoRequest.controller.js';

import { demoRequestSchema } from '../lib/schemas/demo.schema.js';
import { authorizeRoles, protect } from '../middlewares/auth.js';
import { validateInput } from '../middlewares/validate.js';

const demoRequestRouter = express.Router();

demoRequestRouter.post(
  '/',
  validateInput(demoRequestSchema),
  DemoRequestController.Submit
);

demoRequestRouter.get(
  '/',
  protect,
  authorizeRoles('admin'),
  AdminController.fetchDemoRequests
);

export default demoRequestRouter;
