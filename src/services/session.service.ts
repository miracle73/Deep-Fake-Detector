import { v4 as uuidv4 } from 'uuid';
import Session, { ISession } from '../models/Session.js';

export const createSession = async (
  userId: string,
  userAgent: string,
  ipAddress: string,
  deviceId?: string
): Promise<ISession> => {
  await deactivateAllSessions(userId);

  const session = new Session({
    userId,
    sessionId: uuidv4(),
    deviceId,
    userAgent,
    ipAddress,
    isActive: true,
  });

  return await session.save();
};

export const deactivateAllSessions = async (userId: string): Promise<void> => {
  await Session.updateMany(
    {
      userId,
      isActive: true,
    },
    {
      isActive: false,
    }
  );
};

export const deactivateSession = async (
  sessionId: string
): Promise<ISession | null> => {
  return await Session.findByIdAndUpdate(
    { sessionId },
    { isActive: false },
    { new: true }
  );
};

export const activateSession = async (
  sessionId: string,
  userId: string
): Promise<boolean> => {
  const session = await Session.findOne({
    sessionId,
    userId,
    isactive: true,
  });

  if (!session) {
    return false;
  }

  session.lastActive = new Date();
  await session.save();

  return true;
};

export const getActiveSessions = async (
  userId: string
): Promise<ISession[]> => {
  return await Session.find({
    userId,
    isActive: true,
  });
};

export const cleanupExpiredSessions = async (
  maxAgeHours: number = 24
): Promise<void> => {
  const cutoff = new Date();
  cutoff.setHours(cutoff.getHours() - maxAgeHours);

  await Session.updateMany(
    { lastActive: { $lt: cutoff } },
    { isActive: false }
  );
};
