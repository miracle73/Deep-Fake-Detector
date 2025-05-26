import express from 'express';

import * as AuthController from '../controllers/auth.controller.js';
import {
  forgotPasswordSchema,
  loginSchema,
  registerSchema,
} from '../lib/schemas/user.schema.js';
import { validateInput } from '../middlewares/validate.js';

import type { RequestHandler } from 'express';

const router = express.Router();

router.post(
  '/register',
  validateInput(registerSchema),
  AuthController.register as RequestHandler
);

router.post(
  '/login',
  validateInput(loginSchema),
  AuthController.login as RequestHandler
);

router.post(
  '/forgot-password',
  validateInput(forgotPasswordSchema),
  AuthController.forgotPassword as RequestHandler
);

router.post(
  '/reset-password/:resetToken',
  AuthController.resetPassword as RequestHandler
);

export default router;
