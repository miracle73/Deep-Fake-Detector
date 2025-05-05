import express from 'express';

import * as UploadController from '../controllers/upload.controllers.js';
import upload from '../middlewares/upload.js';

const router = express.Router();

router.post('/upload', upload.single('media'), UploadController.uploadSingle);

router.post(
  '/upload/batch',
  upload.array('media', 10),
  UploadController.uploadBatch
);

router.get('/upload/status/:id', UploadController.getStatus);

export default router;
