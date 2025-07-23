import jwt from 'jsonwebtoken';
import mongoose from 'mongoose';

import User from '../../models/User.js';
import { generateToken } from '../../utils/generateToken.js';
import logger from '../../utils/logger.js';
import { api, cleanup, initializeDB, JWT_SECRET, testUser } from '../setup.js';

beforeAll(async () => {
  await initializeDB();
});

afterAll(async () => {
  await cleanup();
});

describe('Subscription Tests', () => {
  describe('Get Subscription Plans', () => {
    describe('given auth token is not provided', () => {
      it('should return 401 Unauthorized', async () => {
        const response = await api
          .get('/api/v1/subscriptions/plans')
          .expect(401);

        logger.info(
          `Fetch plans response: ${JSON.stringify(response, null, 2)}`
        );
      });
    });

    describe('given user does not exist', () => {
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
          .get('/api/v1/subscriptions/plans')
          .set('Authorization', `Bearer ${token}`)
          .expect(401);

        logger.info(
          `Get Subscription Plans Response: ${JSON.stringify(
            response,
            null,
            2
          )}`
        );
      });
    });

    describe('given auth requirement is satisfied', () => {
      it('should return 200 OK', async () => {
        const user = await User.findOne({ email: testUser.email });

        if (!user) {
          throw new Error('Test user not found in the database');
        }

        const token = generateToken(user._id.toString());

        const response = await api
          .get('/api/v1/subscriptions/plans')
          .set('Authorization', `Bearer ${token}`)
          .expect(200);

        logger.info(
          `Get Subscription Plans Response: ${JSON.stringify(
            response,
            null,
            2
          )}`
        );
      });
    });
  });

  describe('Create Checkout Session', () => {
    describe('given auth token is not provided', () => {
      it('should return 401 Unauthorized', async () => {
        const response = await api
          .get('/api/v1/subscriptions/plans')
          .expect(401);

        logger.info(
          `Fetch plans response: ${JSON.stringify(response, null, 2)}`
        );
      });
    });

    describe('given priceId is not provided', () => {
      it('should return 400', async () => {
        const user = await User.findOne({ email: testUser.email });
        if (!user) {
          throw new Error('Test user not found in the database');
        }

        const token = generateToken(user._id.toString());

        const response = await api
          .get('/api/v1/subscriptions/checkout')
          .set('Authorization', `Bearer ${token}`)
          .send({ priceId: 'price_1RXjjyRt69ZCE8YlMDPNod5i' })
          .expect(200);

        logger.info(
          `Create Checkout Session Response: ${JSON.stringify(
            response,
            null,
            2
          )}`
        );
      });
    });
  });
});
