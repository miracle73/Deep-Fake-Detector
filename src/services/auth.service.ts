import jwt from 'jsonwebtoken';
import User from '../models/User.js';

export const generateToken = (
  userId: string,
  sessionVersion: number
): string => {
  return jwt.sign({ userId, sessionVersion }, process.env.JWT_SECRET!, {
    expiresIn: '6h',
  });
};

export const verifyToken = (
  token: string
): { userId: string; sessionVersion: number } => {
  return jwt.verify(token, process.env.JWT_SECRET!) as {
    userId: string;
    sessionVersion: number;
  };
};

export const invalidateAllSessions = async (userId: string): Promise<void> => {
  await User.findByIdAndUpdate(userId, { $inc: { sessionVersion: 1 } });
};

export const validateSession = async (
  userId: string,
  tokenSessionVersion: number
): Promise<boolean> => {
  const user = await User.findById(userId);
  if (!user) return false;

  return user.sessionVersion === tokenSessionVersion;
};
