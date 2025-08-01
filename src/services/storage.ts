import { Storage } from '@google-cloud/storage';
import { config } from 'dotenv';

config();

if (!process.env.GOOGLE_CREDENTIALS_JSON) {
  throw new Error('GOOGLE_CREDENTIALS_JSON environment variable is required');
}

const credentials = JSON.parse(process.env.GOOGLE_CREDENTIALS_JSON);

const storage = new Storage({
  projectId: credentials.project_id,
  credentials,
});

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
