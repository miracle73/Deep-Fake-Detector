import sharp from 'sharp';
import { v4 as uuidv4 } from 'uuid';
import { Readable } from 'node:stream';
import { bucket } from './storage.js';

export async function generateAndUploadThumbnail({
  buffer,
  mimetype,
  originalName,
}: {
  buffer: Buffer;
  mimetype: string;
  originalName: string;
}): Promise<string> {
  const thumbFilename = `thumbnails/${uuidv4()}-${originalName}.jpg`;

  const file = bucket.file(thumbFilename);

  return new Promise((resolve, reject) => {
    const transformer = sharp().resize(300).jpeg({ quality: 80 });

    const uploadStream = file.createWriteStream({
      resumable: false,
      metadata: { contentType: 'image/jpeg' },
    });

    uploadStream.on('error', (err) => {
      console.error('GCS upload error:', err);
      reject(err);
    });

    uploadStream.on('finish', () => {
      console.log('running upload stream');
      resolve(`https://storage.googleapis.com/${bucket.name}/${thumbFilename}`);
    });

    const readable = new Readable();
    readable.push(buffer);
    readable.push(null);

    readable.pipe(transformer).pipe(uploadStream);
  });
}
