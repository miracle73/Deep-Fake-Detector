import nodemailer from 'nodemailer';

import logger from '../utils/logger.js';

interface EmailOptions {
  to: string;
  subject: string;
  html: string;
}

const transporter = nodemailer.createTransport({
  host: process.env.SMTP_HOST,
  secure: true,
  tls: {
    rejectUnauthorized: false,
  },
  requireTLS: true,
  port: Number(process.env.SMTP_PORT),
  auth: {
    user: process.env.SMTP_USER,
    pass: process.env.SMTP_PASS,
  },
});

export async function sendEmail(options: EmailOptions) {
  const from = process.env.SMTP_FROM || 'Safeguard <info@safeguard.com>';
  try {
    const mailOptions = {
      from,
      ...options,
      envelope: {
        from: 'info@safeguardmedia.io',
        to: options.to,
      },
    };

    const result = await transporter.sendMail(mailOptions);

    if (result.rejected.length > 0) {
      logger.error('Email was rejected by recipient server', {
        to: options.to,
        rejected: result.rejected,
        response: result.response,
      });
    } else {
      logger.info(`Email accepted for delivery to ${options.to}`, {
        messageId: result.messageId,
        envelope: result.envelope,
        response: result.response,
      });
    }

    logger.info(`Email sent successfully to ${options.to}`);
  } catch (error) {
    logger.error('Nodemailer send failed', {
      message: error instanceof Error ? error.message : 'Failed to send',
      stack: error instanceof Error && error.stack,
      to: options.to,
    });
    throw new Error('Failed to send email');
  }
}
