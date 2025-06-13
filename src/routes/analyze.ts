import express from 'express';

import * as DetectController from '../controllers/detect.controller.js';
import upload, {
  imageUploadMiddleware,
  multipleImageUploadMiddleware,
} from '../middlewares/upload.js';

const router = express.Router();

router.post('/analyze', imageUploadMiddleware, DetectController.analyze);

router.post(
  '/analyze/batch',
  multipleImageUploadMiddleware,
  DetectController.analyzeBulkMedia
);

router.get('/analyze/status/:id', DetectController.getJobStatus);

export default router;
