import mongoose from 'mongoose';
import logger from '../utils/logger.js';

const connectDB = async () => {
  const MONGODB_URI =
    process.env.NODE_ENV === 'test'
      ? process.env.TEST_MONGODB_URI
      : process.env.MONGODB_URI;

  try {
    if (!MONGODB_URI) {
      throw new Error('MONGODB_URI is not defined');
    }

    const conn = await mongoose.connect(MONGODB_URI);
    logger.info('Connected to database successfully💥', conn.connection.host);
  } catch (error) {
    logger.error('Error connecting to database:', error);
    process.exit(1);
  }
};

export default connectDB;
