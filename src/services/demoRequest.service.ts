import DemoRequest from 'models/DemoRequest';
import { AppError } from 'utils/error';

interface UserData {
  firstName: string;
  lastName: string;
  email: string;
  role: string;
  goal: string;
  contentType?: string;
  urgencyLevel?: string;
  metadata?: string;
}

export const createDemoUser = async (userdata: UserData) => {
  const existingUser = await DemoRequest.findOne({ email: userdata.email });

  if (existingUser) {
    throw new AppError(400, 'Demo user with this email already exists');
  }
  const user = await DemoRequest.create(userdata);

  if (!user) {
    throw new AppError(400, 'Failed to create demo user');
  }

  return user;
};
