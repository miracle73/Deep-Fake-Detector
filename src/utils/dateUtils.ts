import type { IUser } from '../types/user.js';

export function getMonthlyResetDate(date = new Date()): Date {
  // to reset at UTC midnight on 1st of month always
  return new Date(
    Date.UTC(
      date.getUTCFullYear(),
      date.getUTCMonth() + 1, // next month
      1, // first day
      0,
      0,
      0,
      0 // midnight
    )
  );
}

export function getNextResetDate(user: IUser): Date | null {
  if (user.unlimitedQuota) return null;
  return user.usageQuota?.lastResetAt
    ? getMonthlyResetDate(user.usageQuota.lastResetAt)
    : getMonthlyResetDate();
}
