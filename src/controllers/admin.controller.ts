import type { NextFunction, Request, Response } from 'express';
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
