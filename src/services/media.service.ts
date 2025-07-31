import sharp from 'sharp';
import { v4 as uuidv4 } from 'uuid';
import { bucket } from './storage';

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

  const resized = await sharp(buffer)
    .resize(300)
    .jpeg({ quality: 80 })
    .toBuffer();

  console.log('bucket upload starting', thumbFilename);

  const file = bucket.file(thumbFilename);
  const stream = file.createWriteStream({
    resumable: false,
    metadata: { contentType: 'image/jpeg' },
  });

  console.log('igot here so lets fix up');

  return new Promise((resolve, reject) => {
    stream.on('error', (err) => reject(err));
    stream.on('finish', async () => {
      //   await file.makePublic();
      resolve(`https://storage.googleapis.com/${bucket.name}/${thumbFilename}`);
    });
    stream.end(resized);
  });
}

// monitoring-logger@deep-fake-detector-465915.iam.gserviceaccount.com
