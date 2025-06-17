import type { IUser } from './user.js';

declare global {
  namespace Express {
    interface Request {
      user?: IUser;
      // biome-ignore lint/suspicious/noExplicitAny: <explanation>
      validatedQuery?: any;
      // validatedBody?: unknown;
      // validatedParams?: unknown;
    }
  }
}
