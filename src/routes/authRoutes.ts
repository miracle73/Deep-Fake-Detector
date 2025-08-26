import express from 'express';

import * as AuthController from '../controllers/auth.controller.js';
import {
  forgotPasswordSchema,
  loginSchema,
  registerSchema,
  resendVerificationEmailSchema,
  resetPasswordBodySchema,
  verifyEmailQuerySchema,
} from '../lib/schemas/user.schema.js';
import { convertAccessTokenToIdToken, protect } from '../middlewares/auth.js';
import { validateInput, validateQuery } from '../middlewares/validate.js';

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
  validateInput(resetPasswordBodySchema),
  AuthController.resetPassword as RequestHandler
);

router.post(
  '/google',
  convertAccessTokenToIdToken,
  AuthController.googleLogin as RequestHandler
);

router.get(
  '/verify-email',
  validateQuery(verifyEmailQuerySchema),
  AuthController.verifyEmail as RequestHandler
);

router.post(
  '/resend-verification',
  validateInput(resendVerificationEmailSchema),
  AuthController.resendVerificationEmail
);

router.post('/logout', protect, AuthController.logout);

router.get('/create-access-code', AuthController.generateAccessCode);

export default router;
