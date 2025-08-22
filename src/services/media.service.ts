// import { path as ffmpegPath } from '@ffmpeg-installer/ffmpeg';
import ffmpeg from 'fluent-ffmpeg';
import { PassThrough, Readable } from 'node:stream';
import sharp from 'sharp';
import { v4 as uuidv4 } from 'uuid';

import { bucket } from './storage.js';

ffmpeg.setFfmpegPath('/usr/bin/ffmpeg');

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

export async function generateAndUploadVideoThumbnail({
  buffer,
  originalName,
}: {
  buffer: Buffer;
  originalName: string;
}): Promise<string> {
  const thumbFilename = `thumbnails/${uuidv4()}-${originalName}.jpg`;
  const file = bucket.file(thumbFilename);

  return new Promise((resolve, reject) => {
    // Turn the incoming buffer into a stream for ffmpeg
    const inputStream = new PassThrough();
    inputStream.end(buffer);

    // Create upload stream for GCS
    const uploadStream = file.createWriteStream({
      resumable: false,
      metadata: { contentType: 'image/jpeg' },
    });

    uploadStream.on('error', reject);
    uploadStream.on('finish', () => {
      resolve(`https://storage.googleapis.com/${bucket.name}/${thumbFilename}`);
    });

    // Run ffmpeg to grab a frame at 1 second in
    ffmpeg(inputStream)
      .inputFormat('mp4') // if you're sure videos are mp4; else detect dynamically
      .on('error', (err) => {
        console.error('ffmpeg error:', err);
        reject(err);
      })
      .frames(1) // just 1 frame
      .seekInput(1) // at 1 second
      .format('image2') // force image output
      .size('300x300') // resize (like sharp did)
      .outputOptions('-qscale:v 3') // quality (lower is better quality; 2–5 is good)
      .pipe(uploadStream, { end: true });
  });
}
