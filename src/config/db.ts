import mongoose from 'mongoose';
import logger from '../utils/logger.js';

const connectDB = async () => {
  try {
    if (!process.env.MONGODB_URI) {
      throw new Error('MONGODB_URI is not defined');
    }

    const conn = await mongoose.connect(process.env.MONGODB_URI);
    logger.info('Connected to database successfully💥', conn.connection.host);
  } catch (error) {
    logger.error('Error connecting to database:', error);
    process.exit(1);
  }
};

export default connectDB;
