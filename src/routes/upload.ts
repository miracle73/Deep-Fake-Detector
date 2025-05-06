import express from 'express';

import * as UploadController from '../controllers/upload.controllers.js';
import upload, {
  imageUploadMiddleware,
  multipleImageUploadMiddleware,
} from '../middlewares/upload.js';

const router = express.Router();

router.post('/analyze', imageUploadMiddleware, UploadController.analyze);

router.post(
  '/analyze/batch',
  multipleImageUploadMiddleware,
  UploadController.analyzeBulkMedia
);

router.get('/analyze/status/:id', UploadController.getJobStatus);

export default router;
