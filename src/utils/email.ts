import { sendEmail } from '../services/emailService.js';
import { generatePasswordResetEmail } from './email.templates.js';
import logger from './logger.js';

export async function sendPasswordResetEmail(to: string, resetLink: string) {
  const html = generatePasswordResetEmail(resetLink);

  await sendEmail({
    to,
    subject: 'Password Reset Request',
    html,
  });

  logger.info(`Password reset email sent to ${to}`);
}
