import express from 'express';

import * as UploadController from '../controllers/upload.controllers.js';
import upload from '../middlewares/upload.js';

const router = express.Router();

router.post('/analyze', upload.single('media'), UploadController.analyze);

router.post(
  '/analyze/batch',
  upload.array('media', 10),
  UploadController.analyzeBulkMedia
);

router.get('/analyze/status/:id', UploadController.getStatus);

export default router;
