import { Storage } from '@google-cloud/storage';
import fs from 'node:fs';

const storage = new Storage();
const bucketName = process.env.GCS_BUCKET_NAME || 'deepfake-detector-media';

export async function uploadToGCS(file: Express.Multer.File): Promise<string> {
  const destination = `uploads/${Date.now()}_${file.originalname}`;
  const bucket = storage.bucket(bucketName);

  await bucket.upload(file.path, { destination });

  // Optional cleanup
  fs.unlinkSync(file.path);

  // Get public URL or signed URL
  return `https://storage.googleapis.com/${bucketName}/${destination}`;
}
