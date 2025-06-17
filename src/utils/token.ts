import jwt from 'jsonwebtoken';
import type { Secret } from 'jsonwebtoken';

export const generateEmailVerificationToken = (userId: string) => {
  return jwt.sign({ userId }, process.env.EMAIL_VERIFICATION_SECRET as Secret, {
    expiresIn: '1d',
  });
};

export const verifyEmailVerificationToken = (token: string) => {
  return jwt.verify(token, process.env.EMAIL_VERIFICATION_SECRET as Secret) as {
    userId: string;
  };
};
