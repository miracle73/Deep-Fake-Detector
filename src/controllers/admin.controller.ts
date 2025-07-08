import type { NextFunction, Request, Response } from 'express';
import { businessIntegrationSchema } from 'lib/schemas/businessIntegration.schema';
import BusinessIntegration from 'models/BusinessIntegration';
import { getDemoRequests } from 'services/demoRequest.service';
import logger from 'utils/logger';

export const fetchDemoRequests = async (
  req: Request,
  res: Response,
  next: NextFunction
) => {
  try {
    const requests = await getDemoRequests();

    res.status(200).json({
      status: 'success',
      message: 'Demo requests fetched successfully',
      data: requests,
    });
  } catch (error) {
    logger.error(`Error fetching Demo Requests: ${error}`);
    next(error);
  }
};

export const getAllBusinessIntegrations = async (
  req: Request,
  res: Response,
  next: NextFunction
) => {
  try {
    const { page = 1, limit = 10, status } = req.query;

    // biome-ignore lint/suspicious/noExplicitAny: <explanation>
    const filter: Record<string, any> = {};
    if (status) filter.status = status;

    const requests = await BusinessIntegration.find(filter)
      .sort({ createdAt: -1 })
      .skip((Number(page) - 1) * Number(limit))
      .limit(Number(limit));

    const total = await BusinessIntegration.countDocuments(filter);

    res.status(200).json({
      success: true,
      data: requests,
      pagination: {
        page: Number(page),
        limit: Number(limit),
        total,
      },
    });
  } catch (error) {
    next(error);
  }
};

export const getBusinessIntegrationById = async (
  req: Request,
  res: Response,
  next: NextFunction
) => {
  try {
    const { id } = req.params;
    const request = await BusinessIntegration.findById(id);

    if (!request) {
      res.status(404).json({ success: false, error: 'Request not found' });
    }

    res.status(200).json({ success: true, data: request });
  } catch (error) {
    next(error);
  }
};

export const updateBusinessIntegrationStatus = async (
  req: Request,
  res: Response,
  next: NextFunction
) => {
  try {
    const { id } = req.params;
    const { status } = req.body;

    const validation = businessIntegrationSchema
      .pick({ status: true })
      .safeParse({ status });
    if (!validation.success) {
      res.status(400).json({
        success: false,
        error:
          validation.error.flatten().fieldErrors.status?.[0] ||
          'Invalid status',
      });
    }

    const updatedRequest = await BusinessIntegration.findByIdAndUpdate(
      id,
      { status },
      { new: true } // Return the updated document
    );

    if (!updatedRequest) {
      res.status(404).json({ success: false, error: 'Request not found' });
    }

    res.status(200).json({ success: true, data: updatedRequest });
  } catch (error) {
    next(error);
  }
};
