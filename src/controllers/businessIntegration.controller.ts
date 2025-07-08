import emailQueue from '../queues/emailQueue.js';
import { createBusinessIntegrationRequest } from '../services/businessIntegration.service.js';
import logger from '../utils/logger.js';

import type { NextFunction, Request, Response } from 'express';
import type { BusinessIntegrationType } from '../lib/schemas/businessIntegration.schema.js';
export const submitBusinessIntegration = async (
  req: Request,
  res: Response,
  next: NextFunction
) => {
  try {
    const record = await createBusinessIntegrationRequest(
      req.body as BusinessIntegrationType
    );

    await emailQueue.add('businessIntegrationEmail', {
      to: record.email,
      subject: 'Business Integration Request Received',
      html: `<p>Dear ${record.firstName},</p> <p>Thank you for your interest in integrating with our business. We have received your request and will review it shortly.</p><p>Best regards,</p> <p>The Team</p>`,
    });

    res.status(201).json({
      success: true,
      message: 'Business integration request submitted successfully',
      data: record,
    });
  } catch (error) {
    logger.error(error);
    next(error);
  }
};
