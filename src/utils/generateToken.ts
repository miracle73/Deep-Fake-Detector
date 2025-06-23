import jwt from 'jsonwebtoken';

import type { Secret, SignOptions } from 'jsonwebtoken';
import type { ObjectId } from 'mongoose';

export const generateToken = (userId: string | ObjectId) => {
  return jwt.sign(
    { id: userId },
    process.env.JWT_SECRET as Secret,
    {
      expiresIn: process.env.JWT_EXPIRE || '30d',
    } as SignOptions
  );
};
