import { Storage } from '@google-cloud/storage';

const storage = new Storage();

const bucketName = 'deep-fake-001'; // Your bucket
const bucket = storage.bucket(bucketName);

export { bucket };
