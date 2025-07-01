import jwt from 'jsonwebtoken';
import mongoose from 'mongoose';

import User from '../../models/User.js';
import { generateToken } from '../../utils/generateToken.js';
import logger from '../../utils/logger';
import { api, cleanup, initializeDB, JWT_SECRET, testUser } from '../setup';

beforeAll(async () => {
  await initializeDB();
});

afterAll(async () => {
  await cleanup();
});

describe('User Tests', () => {
  describe('Get User', () => {
    describe('given the user does not exist', () => {
      it('should return a 404 error', async () => {
        const userData = {
          userId: new mongoose.Types.ObjectId().toString(),
          email: 'hello@jackhayes.com',
          password: 'password',
          firstName: 'Jack',
          lastName: 'Hayes',
          agreedToTerms: true,
          userType: 'individual',
          termsAgreedAt: new Date(),
        };
        const token = jwt.sign(userData, JWT_SECRET);

        const response = await api
          .get('/api/v1/user/')
          .set('Authorization', `Bearer ${token}`)
          .expect(401);

        // logger.info(
        //   `Get User Response: ${response.body} -:-  ${JSON.stringify(
        //     response,
        //     null,
        //     2
        //   )}.`
        // );
      });
    });

    describe('given user exists', () => {
      it('should retrieve user profile', async () => {
        const user = await User.findOne({ email: testUser.email }).lean();

        if (!user) {
          throw new Error('Test user not found in the database');
        }

        const token = generateToken(user._id.toString());

        const response = await api
          .get('/api/v1/user/')
          .set('Authorization', `Bearer ${token}`)
          .expect(200);

        // logger.info(
        //   `Get User test response: ${JSON.stringify(response.body, null, 2)}`
        // );

        // expect(response.body.data.email).toBe(testUser.email);
        // expect(response.body.data.password).toBeUndefined();
      });
    });
  });

  describe('Update User', () => {
    it('should update user profile', async () => {
      const user = await User.findOne({ email: testUser.email }).lean();

      if (!user) {
        throw new Error('Test user not found in the database');
      }

      const token = generateToken(user._id.toString());

      const response = await api
        .patch('/api/v1/user/update')
        .set('Authorization', `Bearer ${token}`)
        .send({
          firstName: 'Jake',
        })
        .expect(200);

      logger.info(
        'Update User Response: ',
        JSON.stringify(response.body, null, 2)
      );

      // expect(response.body.data.updatedUser.firstName).toBe('Jake');
    });
  });

  describe('Delete User', () => {
    it('should delete user profile', async () => {
      const user = await User.findOne({ email: testUser.email }).lean();

      if (!user) {
        throw new Error('Test user not found in the database');
      }

      const token = generateToken(user._id.toString());

      const response = await api
        .delete('/api/v1/user/delete')
        .set('Authorization', `Bearer ${token}`)
        .expect(200);

      logger.info(
        'Delete User Response: ',
        JSON.stringify(response.body, null, 2)
      );
    });
  });
});
