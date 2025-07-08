import { generateDemoConfirmationEmail } from 'utils/email.templates.js';

import emailQueue from '../queues/emailQueue.js';
import { createDemoUser } from '../services/demoRequest.service.js';
import logger from '../utils/logger.js';

import type { Request, NextFunction, Response } from 'express';
import type { DemoRequest } from '../lib/schemas/demo.schema.js';

export const Submit = async (
  req: Request,
  res: Response,
  next: NextFunction
) => {
  try {
    const user = await createDemoUser(req.body as DemoRequest);

    // add to queue
    // await sendDemoConfirmationEmail(email, name); // basic "you’re on the waitlist" email

    const html = generateDemoConfirmationEmail(user.firstName);

    await emailQueue.add('demoConfirmationEmail', {
      to: user.email,
      subject: 'Demo User Created Successfully',
      html,
    });

    res.status(201).json({
      success: true,
      message: 'Demo user created successfully',
      data: user,
    });
  } catch (error) {
    logger.error('Error in Submit controller:', error);
    next(error);
  }
};
