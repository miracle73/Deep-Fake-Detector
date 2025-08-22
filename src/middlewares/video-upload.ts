import multer from 'multer';

import type { Request, Response, NextFunction } from 'express';

export const MAX_VIDEO_SIZE = 100 * 1024 * 1024; // 100MB

export const ALLOWED_VIDEO_TYPES = [
  'video/mp4',
  'video/quicktime',
  'video/x-msvideo',
];

export const ALLOWED_VIDEO_EXTENSIONS = ['.mp4', '.mov', '.avi'];

export const videoUpload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: MAX_VIDEO_SIZE,
    files: 1,
  },
  fileFilter: (req, file, cb) => {
    if (ALLOWED_VIDEO_TYPES.includes(file.mimetype)) {
      cb(null, true);
    } else {
      cb(new Error('INVALID_FILE_TYPE: Only videos are allowed'));
    }
  },
});

export const videoUploadMiddleware = (
  req: Request,
  res: Response,
  next: NextFunction
) => {
  const upload = videoUpload.single('video');

  // biome-ignore lint/suspicious/noExplicitAny: <explanation>
  upload(req, res, (err: any) => {
    if (err) {
      if (err.code === 'LIMIT_FILE_SIZE') {
        return res.status(413).json({
          success: false,
          error: 'File too large',
          message: `Maximum video size is ${MAX_VIDEO_SIZE / 1024 / 1024}MB`,
          errorCode: 'FILE_TOO_LARGE',
        });
      }

      if (err.message?.startsWith('INVALID_FILE_TYPE')) {
        return res.status(400).json({
          success: false,
          error: 'Invalid file type',
          message: 'Only video files are allowed',
          errorCode: 'INVALID_FILE_TYPE',
          allowedTypes: ALLOWED_VIDEO_TYPES,
        });
      }

      return res.status(500).json({
        success: false,
        error: 'Upload failed',
        message: err.message,
        errorCode: 'UPLOAD_ERROR',
      });
    }

    if (!req.file) {
      return res.status(400).json({
        success: false,
        error: 'No video file provided',
        message: 'Please upload a video file',
        errorCode: 'NO_FILE_PROVIDED',
      });
    }

    next();
  });
};
