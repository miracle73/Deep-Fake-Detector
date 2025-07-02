import User from '../../models/User.js';
import { generateToken } from '../../utils/generateToken.js';
import logger from '../../utils/logger.js';
import { api, cleanup, initializeDB, testUser } from '../setup.js';

beforeAll(async () => {
  await initializeDB();
});

afterAll(async () => {
  await cleanup();
});

describe('Waitlist Test', () => {
  describe('Waitlist Signup', () => {
    describe('given email is not provided', () => {
      it('should return 400 error', async () => {
        const signUpData = {
          email: 'hello@johndoe.com',
        };

        const response = await api
          .post('/api/v1/waitlist/signup')
          .send(signUpData)
          .expect(400);

        logger.info(
          `Waitlist Signup Response: ${JSON.stringify(response, null, 2)}`
        );
      });
    });

    describe('given email already exists in waitlist', () => {
      it('should return 409 Conflict Error', async () => {
        const waitlistSignUpData = {
          email: 'hello@janedoe.com',
        };

        const response = await api
          .post('/api/v1/waitlist/signup')
          .send(waitlistSignUpData)
          .expect(409);

        logger.info(
          `Waitlist Signup Response: ${JSON.stringify(response, null, 2)}`
        );
      });
    });

    describe('given email is provided and signup is successful', () => {
      it('should return 201 OK', async () => {
        const waitlistSignUpData = {
          email: 'hello@johndoe.com',
        };

        const response = await api
          .post('/api/v1/waitlist/signup')
          .send(waitlistSignUpData)
          .expect(201);

        logger.info(
          `Waitlist Signup Response: ${JSON.stringify(response, null, 2)}`
        );
      });
    });
  });

  describe('Get Waitlist Stats', () => {
    describe('given user is not admin', () => {
      it('should return 401: Authentication failed ', async () => {
        const user = await User.findOne({ email: testUser.email });

        if (!user) {
          throw new Error('Test user not found in the database');
        }

        const token = generateToken(user._id.toString());

        const response = await api
          .get('/api/v1/waitlist/stats')
          .set('Authorization', `Bearer ${token}`)
          .expect(401);

        logger.info(
          `Get Waitlist Stats Response: ${JSON.stringify(response, null, 2)}`
        );
      });
    });

    describe('given user is an admin', () => {
      it('should fetch stats successfully', async () => {
        const adminUser = await User.findOne({ role: 'admin' });

        if (!adminUser) {
          throw new Error('Admin user not found in the database');
        }

        const token = generateToken(adminUser._id.toString());

        const response = await api
          .get('/api/v1/waitlist/stats')
          .set('Authorization', `Bearer ${token}`)
          .expect(200);

        logger.info(
          `Get Waitlist Stats Response: ${JSON.stringify(response, null, 2)}`
        );

        expect(response.body.success).toBe(true);
      });
    });
  });
});
