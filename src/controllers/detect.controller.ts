import { PubSub } from '@google-cloud/pubsub';
import axios from 'axios';
import FormData from 'form-data';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { v4 as uuidv4 } from 'uuid';

import User from '../models/User.js';
import { storeAnalysis } from '../services/analysis.media.js';
import {
  detectionJobs,
  simulateDetection,
} from '../services/detectionQueue.js';
import {
  generateAndUploadThumbnail,
  generateAndUploadVideoThumbnail,
} from '../services/media.service.js';
import { callVertexAIBatch } from '../services/vertexClient.js';
import { bucket } from '../utils/gcsClient.js';
import { cloudLogger } from '../utils/google-cloud/logger.js';
import { pushMetric } from '../utils/google-cloud/metrics.js';
import logger from '../utils/logger.js';

import type { NextFunction, Request, Response } from 'express';
import type { Response as ExpressResponse } from 'express';
import type { DetectionJob } from '../services/detectionQueue.js';
import type { AuthRequest } from '../middlewares/auth.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const pubsub = new PubSub();

const MODEL_API_URL =
  process.env.MODEL_API_URL ||
  'https://image-deepfake-detector-production.up.railway.app';

const VIDEO_API_URL =
  process.env.VIDEO_API_URL ||
  'https://video-deepfake-detector-production.up.railway.app';

export const analyze = async (
  req: AuthRequest,
  res: ExpressResponse,
  next: NextFunction
): Promise<void> => {
  try {
    const file = req.file;
    if (!file) {
      res.status(400).json({
        statusCode: 400,
        status: 'error',
        success: false,
        message: 'No file uploaded',
        errorCode: 'NO_FILE',
        details: 'Please upload a valid image file',
      });

      return;
    }

    const thumbnailUrl = await generateAndUploadThumbnail({
      buffer: file.buffer,
      originalName: file.originalname,
      mimetype: file.mimetype,
    });

    console.log('Thumbnail uploaded to:', thumbnailUrl);

    const user = await User.findById(req.user._id);

    const formData = new FormData();
    formData.append('image', file.buffer, {
      filename: file.originalname,
      contentType: file.mimetype,
    });

    const response = await axios.post(`${MODEL_API_URL}/predict`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });

    const { confidence } = response.data;

    await storeAnalysis({
      user,
      confidence,
      file,
      thumbnailUrl,
    });

    await pushMetric({ type: 'detection_count', value: 1 });

    await cloudLogger.info({
      message: `User ${req.user._id} performed image detection`,
    });

    res.status(200).json({
      success: true,
      message: 'Image analysis complete',
      data: response.data,
    });
  } catch (error) {
    logger.info('Failed to upload image', error);
    if (error instanceof Error) {
      res.status(500).json({ error: error.message });
      return;
    }
    res.status(500).json({ error: 'Unknown error occurred' });
    return;
  }
};

export const analyzeBulkMedia = async (
  req: Request,
  res: Response,
  next: NextFunction
): Promise<void> => {
  try {
    const files = req.files as Express.Multer.File[];

    if (!files || files.length === 0) {
      res.status(400).json({
        success: false,
        errorCode: 'NO_FILES',
        message: 'No files uploaded',
        details: 'Please upload at least one file',
      });
      return;
    }

    // Validate all files first
    for (const file of files) {
      if (!file.buffer || file.size === 0) {
        res.status(400).json({
          success: false,
          errorCode: 'INVALID_FILE',
          message: 'One or more files are invalid',
          details: `File ${file.originalname} is empty or corrupted`,
        });

        return;
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

    const mediaUrls = results.map((result) => result.fileInfo.publicUrl);

    const predictions = await callVertexAIBatch(mediaUrls);

    res.status(202).json({
      // 202 Accepted for async processing
      success: true,

      message: 'Bulk analysis started',

      count: results.length,
      jobs: results,
      data: {
        predictions: predictions.map(
          (
            prediction: {
              isDeepfake: boolean;
              confidence: number;
              message: string;
            },
            index: number
          ) => ({
            jobId: results[index].jobId,
            isDeepfake: prediction.isDeepfake,
            confidence: prediction.confidence,
            message: prediction.message,
            fileInfo: results[index].fileInfo,
          })
        ),
        // Add more metadata as needed
      },
      links: {
        status: `/jobs/${results[0].jobId}/status`,
        results: `/jobs/${results[0].jobId}/results`,
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

    return;
  }
};

export const analyzeVideo = async (
  req: AuthRequest,
  res: Response,
  next: NextFunction
): Promise<void> => {
  const file = req.file;
  if (!file) {
    res.status(400).json({
      statusCode: 400,
      status: 'error',
      success: false,
      message: 'No file uploaded',
      errorCode: 'NO_FILE',
      details: 'Please upload a valid image file',
    });

    return;
  }

  const thumbnailUrl = await generateAndUploadVideoThumbnail({
    buffer: file.buffer,
    originalName: file.originalname,
  });

  const user = await User.findById(req.user._id);

  const formData = new FormData();
  formData.append('video', file.buffer, {
    filename: file.originalname,
    contentType: file.mimetype,
  });

  const response = await axios.post(
    `${VIDEO_API_URL}/analyze_temporal`,
    formData,
    {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    }
  );

  const { confidence } = response.data;

  await storeAnalysis({
    user,
    confidence,
    file,
    thumbnailUrl,
  });

  await pushMetric({ type: 'detection_count', value: 1 });

  await cloudLogger.info({
    message: 'Video analysis complete',
    context: {
      userId: req.user._id,
      thumbnailUrl,
      confidence,
      analysisId: response.data.analysisId,
    },
  });

  logger.info('Video analysis complete', {
    userId: req.user._id,
    thumbnailUrl,
    confidence,
  });

  // // Publish to Pub/Sub topic
  // const topicName = 'video-analysis-complete';
  // const dataBuffer = Buffer.from(
  //   JSON.stringify({
  //     userId: req.user._id,
  //     thumbnailUrl,
  //     confidence,
  //     analysisId: response.data.analysisId,
  //   })
  // );
  // await pubsub.topic(topicName).publish(dataBuffer);
  // logger.info('Published video analysis complete message to Pub/Sub', {
  //   topicName,
  //   userId: req.user._id,
  //   thumbnailUrl,
  //   confidence,
  //   analysisId: response.data.analysisId,
  // });

  if (!response.data) {
    res.status(500).json({
      success: false,
      error: 'Video analysis failed',
      message: 'No data returned from model API',
      errorCode: 'MODEL_API_ERROR',
    });
    return;
  }
  if (response.data.error) {
    res.status(500).json({
      success: false,
      error: 'Video analysis failed',
      message: response.data.error,
      errorCode: 'MODEL_API_ERROR',
    });
    return;
  }

  res.status(200).json({
    success: true,
    message: 'Video analysis complete',
    thumbnailUrl,
    data: response.data,
  });
};

export const getStatus = async (req: Request, res: Response) => {
  const job = detectionJobs.get(req.params.id);
  if (!job) res.status(404).json({ message: 'Job not found' });

  res.status(200).json({ id: req.params.id, ...job });
};

export const getJobStatus = (req: Request, res: Response): Promise<void> => {
  const { jobId } = req.params;
  const job = detectionJobs.get(jobId);

  if (!job) {
    res.status(404).json({
      success: false,
      errorCode: 'JOB_NOT_FOUND',
      message: 'Job not found or expired',
    });
    return Promise.resolve();
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

  return Promise.resolve();
};
