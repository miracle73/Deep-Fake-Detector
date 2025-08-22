import multer from 'multer';
import { v4 as uuidv4 } from 'uuid';

import type { NextFunction, Request, Response } from 'express';

// Supported image MIME types
const ALLOWED_IMAGE_TYPES = [
  'image/jpeg', // .jpg, .jpeg
  'image/png', // .png
  'image/webp', // .webp
  'image/gif', // .gif
  'image/tiff', // .tiff
  'image/svg+xml', // .svg
];

interface MulterErrorWithCode extends Error {
  code?: string;
}

// Maximum file size (5MB)
const MAX_FILE_SIZE = 5 * 1024 * 1024;

// Custom file naming function
const generateFilename = (req: Request, file: Express.Multer.File) => {
  const timestamp = Date.now();
  const randomString = uuidv4().substring(0, 8);
  const originalName = file.originalname.replace(/\s+/g, '-').toLowerCase();
  return `img-${timestamp}-${randomString}-${originalName}`;
};

const imageUpload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: MAX_FILE_SIZE,
    files: 1,
  },
  fileFilter: (req, file, cb) => {
    // Validate file type
    if (!ALLOWED_IMAGE_TYPES.includes(file.mimetype)) {
      const error: MulterErrorWithCode = new Error(
        `Invalid file type. Supported types: ${ALLOWED_IMAGE_TYPES.join(', ')}`
      );
      error.code = 'INVALID_FILE_TYPE';
      return cb(error);
    }

    cb(null, true);
  },
});

export const imageUploadMiddleware = (
  req: Request,
  res: Response,
  next: NextFunction
) => {
  const upload = imageUpload.single('image'); // Field name is 'image'

  // biome-ignore lint/suspicious/noExplicitAny: <explanation>
  upload(req, res, (err: any) => {
    if (err) {
      if (err.code === 'LIMIT_FILE_SIZE') {
        return res.status(413).json({
          success: false,
          error: 'File too large',
          message: `Maximum file size is ${MAX_FILE_SIZE / 1024 / 1024}MB`,
          errorCode: 'FILE_TOO_LARGE',
        });
      }

      if (err.code === 'INVALID_FILE_TYPE') {
        return res.status(400).json({
          success: false,
          error: 'Invalid file type',
          message: err.message,
          errorCode: 'INVALID_FILE_TYPE',
          allowedTypes: ALLOWED_IMAGE_TYPES,
        });
      }

      return res.status(500).json({
        success: false,
        error: 'Upload failed',
        message: err.message,
        errorCode: 'UPLOAD_ERROR',
      });
    }

    next();
  });
};

export default imageUpload;

export const multipleImageUploadMiddleware = (
  req: Request,
  res: Response,
  next: NextFunction
) => {
  const upload = imageUpload.array('media', 10); // 'media' field name, max 10 files

  // biome-ignore lint/suspicious/noExplicitAny: <explanation>
  upload(req, res, (err: any) => {
    if (err) {
      if (err.code === 'LIMIT_FILE_SIZE') {
        return res.status(413).json({
          success: false,
          error: 'File too large',
          message: `Maximum file size is ${
            MAX_FILE_SIZE / 1024 / 1024
          }MB per file`,
          errorCode: 'FILE_TOO_LARGE',
        });
      }
      if (err.code === 'INVALID_FILE_TYPE') {
        return res.status(400).json({
          success: false,
          error: 'Invalid file type',
          message: err.message,
          errorCode: 'INVALID_FILE_TYPE',
          allowedTypes: ALLOWED_IMAGE_TYPES,
        });
      }

      if (err.code === 'LIMIT_FILE_COUNT') {
        return res.status(400).json({
          success: false,
          error: 'Too many files',
          message: 'Maximum 10 files allowed per upload',
          errorCode: 'TOO_MANY_FILES',
        });
      }

      return res.status(500).json({
        success: false,
        error: 'Upload failed',
        message: err.message,
        errorCode: 'UPLOAD_ERROR',
      });
    }

    if (!req.files || req.files.length === 0) {
      return res.status(400).json({
        success: false,
        error: 'No files uploaded',
        message: 'Please upload at least one file',
        errorCode: 'NO_FILES',
      });
    }

    next();
  });
};
