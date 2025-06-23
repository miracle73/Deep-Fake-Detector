import express from 'express';

import { protect } from '../middlewares/auth.js';

import * as DetectController from '../controllers/detect.controller.js';
import upload, {
  imageUploadMiddleware,
  multipleImageUploadMiddleware,
} from '../middlewares/upload.js';
import { validateAndDecrementQuota } from '../utils/detect.js';

const router = express.Router();

router.post(
  '/analyze',
  protect,
  validateAndDecrementQuota,
  imageUploadMiddleware,
  DetectController.analyze
);

router.post(
  '/analyze/batch',
  protect,
  validateAndDecrementQuota,
  multipleImageUploadMiddleware,
  DetectController.analyzeBulkMedia
);

router.get('/analyze/status/:id', DetectController.getJobStatus);

export default router;
