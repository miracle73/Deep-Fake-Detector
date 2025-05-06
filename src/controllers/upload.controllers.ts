import { v4 as uuidv4 } from 'uuid';

import {
  detectionJobs,
  simulateDetection,
} from '../services/detectionQueue.js';
import { bucket } from '../utils/gcsClient.js';

// import { simulateDetection } from '../services/detectionQueue.js';
import type { NextFunction, Request, Response } from 'express';
import type { Response as ExpressResponse } from 'express';
import type { DetectionJob } from '../services/detectionQueue.js';

export const analyze = async (
  req: Request,
  res: ExpressResponse,
  next: NextFunction
) => {
  try {
    const file = req.file;

    if (!file) {
      return res.status(400).json({
        statusCode: 400,
        status: 'error',
        success: false,
        message: 'No file uploaded',
        errorCode: 'NO_FILE',
        details: 'Please upload a valid image file',
      });
    }

    const filename = `${uuidv4()}-${file.originalname}`;
    const blob = bucket.file(filename);

    const blobStream = blob.createWriteStream({
      resumable: false,
      metadata: {
        contentType: file.mimetype,
        metadata: {
          originalName: file.originalname,
          uploadedAt: new Date().toISOString(),
        },
      },
    });

    await new Promise((resolve, reject) => {
      blobStream.on('error', (err) => {
        console.error('Upload error:', err);
        reject(new Error('Failed to upload file to storage'));
      });

      blobStream.on('finish', resolve);
      blobStream.end(file.buffer);
    });
    const publicUrl = `https://storage.googleapis.com/${bucket.name}/${blob.name}`;

    return res.status(200).json({
      statusCode: 200,
      status: 'success',
      success: true,
      data: {
        uploadedTo: publicUrl,
        result: {
          isDeepfake: true,
          confidence: '94%',
          message: 'Mock detection complete',
          // Add more metadata as needed
          fileInfo: {
            originalName: file.originalname,
            size: file.size,
            mimetype: file.mimetype,
          },
        },
      },
    });
  } catch (error) {
    console.log('Failed to upload image', error);
    if (error instanceof Error) {
      return res.status(500).json({ error: error.message });
    }
    return res.status(500).json({ error: 'Unknown error occurred' });
  }
};

export const analyzeBulkMedia = async (
  req: Request,
  res: Response,
  next: NextFunction
) => {
  try {
    const files = req.files as Express.Multer.File[];

    if (!files || files.length === 0) {
      return res.status(400).json({
        success: false,
        errorCode: 'NO_FILES',
        message: 'No files uploaded',
        details: 'Please upload at least one file',
      });
    }

    // Validate all files first
    for (const file of files) {
      if (!file.buffer || file.size === 0) {
        return res.status(400).json({
          success: false,
          errorCode: 'INVALID_FILE',
          message: 'One or more files are invalid',
          details: `File ${file.originalname} is empty or corrupted`,
        });
      }
    }

    const uploadPromises = files.map(async (file) => {
      const jobId = uuidv4();
      const filename = `${uuidv4()}-${file.originalname}`;
      const blob = bucket.file(filename);

      // Create job entry
      const job: DetectionJob = {
        id: jobId,
        status: 'pending',
        fileInfo: {
          originalName: file.originalname,
          size: file.size,
          mimetype: file.mimetype,
          storageUrl: `gs://${bucket.name}/${filename}`,
          publicUrl: `https://storage.googleapis.com/${bucket.name}/${filename}`,
        },
        createdAt: new Date().toISOString(),
      };
      detectionJobs.set(jobId, job);

      try {
        // Upload file
        await new Promise((resolve, reject) => {
          const blobStream = blob.createWriteStream({
            resumable: false,
            metadata: {
              contentType: file.mimetype,
              metadata: {
                originalName: file.originalname,
                jobId,
                uploadedAt: new Date().toISOString(),
              },
            },
          });

          blobStream.on('error', reject);
          blobStream.on('finish', resolve);
          blobStream.end(file.buffer);
        });

        // Update job status and start processing
        detectionJobs.set(jobId, { ...job, status: 'processing' });
        simulateDetection(jobId);

        return {
          jobId,
          status: 'processing',
          fileInfo: job.fileInfo,
          links: {
            status: `/jobs/${jobId}/status`,
            results: `/jobs/${jobId}/results`,
          },
        };
      } catch (error) {
        // Handle upload errors
        const errorMessage =
          error instanceof Error ? error.message : 'Upload failed';
        detectionJobs.set(jobId, {
          ...job,
          status: 'failed',
          error: errorMessage,
        });

        return {
          jobId,
          status: 'failed',
          error: errorMessage,
          fileInfo: job.fileInfo,
        };
      }
    });

    const results = await Promise.all(uploadPromises);

    res.status(202).json({
      // 202 Accepted for async processing
      success: true,
      count: results.length,
      jobs: results,
      _links: {
        batchStatus: '/jobs/batch/status', // Endpoint to check all jobs status
      },
    });
  } catch (error) {
    console.error('Bulk analysis error:', error);

    res.status(500).json({
      success: false,
      error: 'Bulk analysis failed',
      message: error instanceof Error ? error.message : 'Unknown error',
      errorCode: 'BULK_ANALYSIS_ERROR',
    });
  }
};

export const getStatus = async (req: Request, res: Response) => {
  const job = detectionJobs.get(req.params.id);
  if (!job) res.status(404).json({ message: 'Job not found' });

  res.status(200).json({ id: req.params.id, ...job });
};

export const getJobStatus = (req: Request, res: Response) => {
  const { jobId } = req.params;
  const job = detectionJobs.get(jobId);

  if (!job) {
    return res.status(404).json({
      success: false,
      errorCode: 'JOB_NOT_FOUND',
      message: 'Job not found or expired',
    });
  }

  res.json({
    success: true,
    jobId,
    status: job.status,
    ...(job.status === 'completed' && { result: job.result }),
    ...(job.status === 'failed' && { error: job.error }),
    createdAt: job.createdAt,
    ...(job.completedAt && { completedAt: job.completedAt }),
  });
};
