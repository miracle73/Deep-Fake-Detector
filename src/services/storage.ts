import { Storage } from '@google-cloud/storage';
import { config } from 'dotenv';

config();

if (!process.env.GOOGLE_APPLICATION_CREDENTIALS) {
  throw new Error('GOOGLE_APPLICATION_CREDENTIALS env var is required');
}

const storage = new Storage();

export const bucket = storage.bucket('deepfake-backend-bucket');

export const uploadToGCS = async (
  localPath: string,
  destinationPath: string,
  contentType?: string
): Promise<string> => {
  console.log(bucket.name);
  await bucket.upload(localPath, {
    destination: destinationPath,
    resumable: false,
    metadata: {
      contentType,
    },
  });

  return `https://storage.googleapis.com/${bucket.name}/${destinationPath}`;
};
