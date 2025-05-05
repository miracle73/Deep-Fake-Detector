import {
  detectionJobs,
  simulateDetection,
} from '../services/detectionQueue.js';
import { v4 as uuidv4 } from 'uuid';

import { bucket } from '../utils/gcsClient.js';

import type { NextFunction, Request, Response } from 'express';

export const uploadSingle = async (
  req: Request,
  res: Response,
  next: NextFunction
): Promise<void> => {
  try {
    const file = req.file;
    if (!file) {
      res.status(400).json({ success: false, message: 'No file uploaded' });
    }

    const blob = bucket.file(`${uuidv4()}-${file?.originalname}`);
    const blobStream = blob.createWriteStream({
      resumable: false,
      contentType: file?.mimetype,
    });

    blobStream.on('error', (err) =>
      res.status(500).json({ error: err.message })
    );

    blobStream.on('finish', () => {
      const publicUrl = `https://storage.googleapis.com/${bucket.name}/${blob.name}`;
      res.status(200).json({
        success: true,
        uploadedTo: publicUrl,
        result: {
          isDeepfake: true,
          confidence: '94%',
          message: 'Mock detection complete',
        },
      });
    });

    blobStream.end(file?.buffer);
  } catch (error) {
    console.log('Failed to upload image', error);
    if (error instanceof Error) {
      res.status(500).json({ error: error.message });
    }
    res.status(500).json({ error: 'Unknown error occurred' });
  }
};

export const uploadBatch = async (
  req: Request,
  res: Response,
  next: NextFunction
) => {
  try {
    const files = req.files as Express.Multer.File[];

    if (!files || files.length === 0) {
      res.status(400).json({ message: 'No files uploaded' });
    }

    const uploads = await Promise.all(
      files.map(async (file) => {
        const jobId = uuidv4(); // Unique ID for tracking this job
        const blob = bucket.file(`${uuidv4()}-${file.originalname}`);
        const blobStream = blob.createWriteStream({
          resumable: false,
          contentType: file.mimetype,
        });

        await new Promise((resolve, reject) => {
          blobStream.on('error', reject);
          blobStream.on('finish', resolve);
          blobStream.end(file.buffer);
        });

        const publicUrl = `https://storage.googleapis.com/${bucket.name}/${blob.name}`;

        // Register job as pending and simulate detection
        detectionJobs.set(jobId, { status: 'pending' });
        simulateDetection(jobId);

        return {
          id: jobId,
          uploadedTo: publicUrl,
          status: 'pending',
        };
      })
    );

    res.status(200).json({
      success: true,
      count: uploads.length,
      uploads,
    });
  } catch (error) {
    console.log('Failed to upload batch images', error);

    if (error instanceof Error) {
      res.status(500).json({ error: error.message });
    }
    res.status(500).json({ error: 'Unknown error occurred' });
  }
};

export const getStatus = async (req: Request, res: Response) => {
  const job = detectionJobs.get(req.params.id);
  if (!job) res.status(404).json({ message: 'Job not found' });

  res.status(200).json({ id: req.params.id, ...job });
};
