import nodemailer from 'nodemailer';
import logger from '../utils/logger.js';

interface EmailOptions {
  to: string;
  subject: string;
  text: string;
  html?: string;
}
const transporter = nodemailer.createTransport({
  service: 'smtp',
  host: process.env.SMTP_HOST,
  port: Number(process.env.SMTP_PORT),
  auth: {
    user: process.env.SMTP_USER,
    pass: process.env.SMTP_PASS,
  },
});

export async function sendEmail(options: EmailOptions) {
  try {
    const mailOptions = {
      from: process.env.EMAIL_FROM,
      ...options,
    };

    await transporter.sendMail(mailOptions);
    logger.info(`Email sent successfully to ${options.to}`);
  } catch (error) {
    logger.error('Error sending email:', error);
    throw new Error('Failed to send email');
  }
}
