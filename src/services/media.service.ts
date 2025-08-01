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
  console.log('this is bucket name', bucket.name);
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

  //   let resized: Buffer;

  //   try {
  //     resized = await sharp(buffer).resize(300).jpeg({ quality: 80 }).toBuffer();

  //     if (!resized || !Buffer.isBuffer(resized)) {
  //       throw new Error('Thumbnail buffer is invalid or empty');
  //     }
  //   } catch (err) {
  //     console.error('Sharp resize failed:', err);
  //     throw new Error('Failed to generate thumbnail');
  //   }

  //   const resized = await sharp(buffer)
  //     .resize(300)
  //     .jpeg({ quality: 80 })
  //     .toBuffer();

  //   console.log('this is resized:', resized);
  //   console.log('bucket upload starting', thumbFilename);

  //   const file = bucket.file(thumbFilename);
  //   const stream = file.createWriteStream({
  //     resumable: false,
  //     metadata: { contentType: 'image/jpeg' },
  //   });

  //   console.log('igot here so lets fix up');

  //   return new Promise((resolve, reject) => {
  //     stream.on('error', (err) => reject(err));
  //     stream.on('finish', async () => {
  //       //   await file.makePublic();
  //       resolve(`https://storage.googleapis.com/${bucket.name}/${thumbFilename}`);
  //     });
  //     stream.end(resized);
  //   });

  //   return new Promise((resolve, reject) => {
  //     const stream = file.createWriteStream({
  //       resumable: false,
  //       metadata: { contentType: 'image/jpeg' },
  //     });

  //     stream.on('error', (err) => {
  //       console.error('GCS stream error:', err);
  //       reject(err);
  //     });

  //     stream.on('finish', () => {
  //       resolve(`https://storage.googleapis.com/${bucket.name}/${thumbFilename}`);
  //     });

  //     stream.end(resized);
  //   });
}

// monitoring-logger@deep-fake-detector-465915.iam.gserviceaccount.com
