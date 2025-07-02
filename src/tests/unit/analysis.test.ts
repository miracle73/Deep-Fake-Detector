import supertest from 'supertest';
import mongoose from 'mongoose';
import path from 'node:path';
import fs from 'node:fs';
import { api, cleanup, initializeDB, testUser } from '../setup.js';
import User from '../../models/User.js';
import { generateToken } from '../../utils/generateToken.js';

const imagePath = path.join(__dirname, '../fixtures/sample.jpg');

beforeAll(async () => {
  await initializeDB();
});

afterAll(async () => {
  await cleanup();
});

describe('Analyze Endpoint', () => {
  let token: string;

  beforeAll(async () => {
    const user = await User.findOne({ email: testUser.email });
    if (!user) throw new Error('Test user not found');
    token = generateToken(user._id.toString());
  });

  describe('when no file is uploaded', () => {
    it('should return 400 error', async () => {
      const response = await api
        .post('/api/v1/detect/analyze')
        .set('Authorization', `Bearer ${token}`)
        .expect(400);

      expect(response.body.message).toMatch(/no file/i);
    });
  });

  describe('when file type is invalid', () => {
    it('should return 400 error for non-image', async () => {
      const txtPath = path.join(__dirname, '../fixtures/sample.txt');
      fs.writeFileSync(txtPath, 'This is a text file');

      const response = await api
        .post('/api/v1/detect/analyze')
        .set('Authorization', `Bearer ${token}`)
        .attach('image', txtPath)
        .expect(400);

      expect(response.body.errorCode).toBe('INVALID_FILE_TYPE');

      fs.unlinkSync(txtPath);
    });
  });

  describe('when image is uploaded correctly', () => {
    it('should analyze and return result', async () => {
      if (!fs.existsSync(imagePath)) {
        throw new Error('Sample image file not found');
      }

      const response = await api
        .post('/api/v1/detect/analyze')
        .set('Authorization', `Bearer ${token}`)
        .attach('image', imagePath)
        .expect(200);

      expect(response.body.success).toBe(true);
      expect(response.body.data.result).toHaveProperty('isDeepfake');
      expect(response.body.data.result).toHaveProperty('confidence');
      expect(response.body.data.result).toHaveProperty('message');
    });
  });

  describe('when quota is exhausted', () => {
    it.skip('should return 400 error for exhausted quota', async () => {
      // You would mock the `validateAndDecrementQuota` middleware here
      // or simulate a user with remainingAnalysis = 0
    });
  });
});
