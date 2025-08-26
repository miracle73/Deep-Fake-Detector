import AccessCode, { IAccessCode } from '../models/AccessCode.js';

export async function validateAccessCode(code: string): Promise<{
  isValid: boolean;
  message?: string;
  accessCode?: IAccessCode;
}> {
  try {
    const accessCode = await AccessCode.findOne({
      code: code.toUpperCase().trim(),
      used: false,
    });

    if (!accessCode) {
      return { isValid: false, message: 'Invalid or already used access code' };
    }

    if (accessCode.expiresAt < new Date()) {
      return { isValid: false, message: 'Access code has expired' };
    }

    return { isValid: true, accessCode };
  } catch (error) {
    throw new Error('Error validating access code');
  }
}

export async function markAccessCodeAsUsed(
  code: string,
  userId: string
): Promise<void> {
  await AccessCode.findOneAndUpdate(
    { code: code.toUpperCase().trim() },
    {
      used: true,
      usedBy: userId,
      usedAt: new Date(),
    }
  );
}

export async function createAccessCode(
  code: string,
  expiresInHours: number = 24
): Promise<IAccessCode> {
  const expiresAt = new Date();
  expiresAt.setHours(expiresAt.getHours() + expiresInHours);

  return await AccessCode.create({
    code: code.toUpperCase().trim(),
    expiresAt,
  });
}
