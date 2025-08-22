import express from 'express';

import { protect } from '../middlewares/auth.js';

import * as DetectController from '../controllers/detect.controller.js';
import upload, {
  imageUploadMiddleware,
  multipleImageUploadMiddleware,
} from '../middlewares/upload.js';
import { validateAndDecrementQuota } from '../utils/detect.js';
import { videoUploadMiddleware } from '../middlewares/video-upload.js';

const router = express.Router();

router.post(
  '/analyze',
  protect,
  imageUploadMiddleware,
  // validateAndDecrementQuota,
  DetectController.analyze
);

router.post(
  '/analyze/batch',
  protect,
  validateAndDecrementQuota,
  multipleImageUploadMiddleware,
  DetectController.analyzeBulkMedia
);

router.post(
  '/analyze-video',
  protect,
  videoUploadMiddleware,
  // validateAndDecrementQuota,
  DetectController.analyzeVideo
);

router.get('/analyze/status/:id', DetectController.getJobStatus);

export default router;
