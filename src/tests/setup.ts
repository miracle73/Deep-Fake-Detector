import mongoose from 'mongoose';
import { redisConnection } from '../config/redis.js';
import User from '../models/User.js';
import app from '../index.js';
import supertest from 'supertest';
import { config } from 'dotenv';

config();

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

const testAdmin = {
  email: 'hello@admin.com',
  password: 'password',
  firstName: 'Admin',
  lastName: 'User',
  agreedToTerms: true,
  userType: 'individual',
  termsAgreedAt: new Date(),
  role: 'admin',
};

const JWT_SECRET: string = process.env.JWT_SECRET ? process.env.JWT_SECRET : '';

async function initializeDB() {
  await mongoose.connection.dropDatabase();
  await User.create(testUser);
  await User.create(testAdmin);
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

export { api, testUser, initializeDB, cleanup, JWT_SECRET };
