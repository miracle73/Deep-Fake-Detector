import mongoose from 'mongoose';
import { redisConnection } from '../config/redis.js';
import User from '../models/User.js';
import app from '../index.js';
import supertest from 'supertest';

const api = supertest(app);

const testUser = {
  email: 'hello@janedoe.com',
  password: 'password',
  firstName: 'Jane',
  lastName: 'Doe',
  agreedToTerms: true,
  userType: 'individual',
  termsAgreedAt: new Date(),
};

async function initializeDB() {
  await mongoose.connection.dropDatabase();
  await User.create(testUser);
}

async function cleanup() {
  try {
    await mongoose.connection.close();

    await new Promise((resolve) => setTimeout(resolve, 100));

    await redisConnection.quit();
  } catch (error) {
    console.error('Error during cleanup:', error);
  }
}

export { api, testUser, initializeDB, cleanup };
