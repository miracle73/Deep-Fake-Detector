import User from '../models/User.js';
import DemoRequest from '../models/DemoRequest.js';
import { AppError } from '../utils/error.js';
import emailQueue from '../queues/emailQueue.js';
import { generateEarlyAccessLiveEmail } from '../utils/email.templates.js';
import { generateEmailVerificationToken } from '../utils/token.js';

interface UserData {
  firstName: string;
  lastName: string;
  email: string;
  role: string;
  // goal: string;
  // contentType?: string;
  // urgencyLevel?: string;
  // metadata?: string;
}

export const createDemoUser = async (userdata: UserData) => {
  const existingDemoUser = await DemoRequest.findOne({ email: userdata.email });

  if (existingDemoUser) {
    throw new AppError(400, 'Demo user with this email already exists');
  }

  const existingUser = await User.findOne({ email: userdata.email });

  if (existingUser) {
    throw new AppError(
      400,
      'User exists already. Please sign in with email and password'
    );
  }

  const user = await DemoRequest.create(userdata);

  if (!user) {
    throw new AppError(400, 'Failed to create demo user');
  }

  const emailToken = generateEmailVerificationToken(user._id.toString());

  const activationUrl = `${process.env.FRONTEND_URL}/create-password?email=${userdata.email}&token=${emailToken}`;

  const createPasswordEmail = generateEarlyAccessLiveEmail({
    name: userdata.firstName,
    activationUrl,
  });

  await emailQueue.add('verification-email', {
    to: userdata.email,
    subject: 'Welcome to SafeGuard Media – Create your password',
    html: createPasswordEmail,
  });

  return user;
};

export const getDemoRequests = async () => {
  const demoRequests = await DemoRequest.find()
    .sort({ createdAt: -1 })
    .select('-__v')
    .lean();

  if (!demoRequests || demoRequests.length === 0) {
    return [];
  }

  return demoRequests;
};
