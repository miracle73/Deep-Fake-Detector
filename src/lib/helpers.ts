import { sendEmail } from '../services/emailService';

export function calculateEstimatedWaitTime(position: number): string {
  // assuming 10 users invited per week
  const usersPerWeek = 10;
  const weeksToWait = Math.ceil(position / usersPerWeek);

  if (weeksToWait <= 1) {
    return 'Less than a week';
  }

  if (weeksToWait <= 4) {
    return `${weeksToWait} weeks`;
  }

  const months = Math.ceil(weeksToWait / 4);
  return `${months} month${months > 1 ? 's' : ''}`;
}

export async function sendWelcomeEmail(data: {
  email: string;
  position: number;
  estimatedWaitTime: string;
}) {
  const emailContent = `
  Welcome to our waitlist!
  
  Thank you for your interest in our platform. You've successfully joined our waitlist.
  
  Your Details:
  - Position: #${data.position}
  - Estimated wait time: ${data.estimatedWaitTime}
  - Signup date: ${new Date().toLocaleDateString()}
  
  We'll notify you as soon as a spot becomes available. In the meantime, follow us on social media for updates and behind-the-scenes content.
  
  Thank you for your patience!
  
  Best regards,
  The Team
    `;

  await sendEmail({
    to: data.email,
    subject: `Welcome to the waitlist - Position #${data.position}`,
    text: emailContent,
  });
}

export async function sendInvitationEmail(email: string) {
  const emailContent = `
  Great news! Your invitation is ready!
  
  You've been selected from our waitlist and can now access our platform.
  
  Click the link below to get started:
  [Your invitation link here]
  
  This invitation expires in 48 hours, so be sure to claim your spot soon.
  
  Welcome aboard!
  
  Best regards,
  The Team
    `;

  await sendEmail({
    to: email,
    subject: 'Your invitation is ready! 🎉',
    text: emailContent,
  });
}
