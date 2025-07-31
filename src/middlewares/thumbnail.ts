import path from 'node:path';
import fs from 'node:fs';
import sharp from 'sharp';
import ffmpeg from 'fluent-ffmpeg';
import { uploadToGCS } from '../services/storage.js';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

type ThumbnailOptions = {
  buffer: Buffer;
  originalName: string;
  mimetype: string;
};

export const generateAndUploadThumbnail = async ({
  buffer,
  originalName,
  mimetype,
}: ThumbnailOptions): Promise<string> => {
  const ext = path.extname(originalName);
  const baseName = path.basename(originalName, ext);
  const thumbName = `${baseName}-thumb.jpg`;
  const tempDir = path.join(__dirname, '../temp');

  if (!fs.existsSync(tempDir)) {
    fs.mkdirSync(tempDir, { recursive: true });
  }

  const thumbPath = path.join(tempDir, thumbName);

  if (mimetype.startsWith('image/')) {
    await sharp(buffer).resize(320).jpeg({ quality: 80 }).toFile(thumbPath);
  } else if (mimetype.startsWith('video/')) {
    const tempVideoPath = path.join(tempDir, `${baseName}${ext}`);
    fs.writeFileSync(tempVideoPath, buffer);

    await new Promise<void>((resolve, reject) => {
      ffmpeg(tempVideoPath)
        .on('end', () => {
          fs.unlinkSync(tempVideoPath);
          resolve();
        })
        .on('error', (err) => {
          reject(err);
        })
        .screenshots({
          timestamps: ['1'],
          filename: thumbName,
          folder: tempDir,
          size: '320x?',
        });
    });
  } else {
    throw new Error('Unsupported file type for thumbnail generation');
  }

  // Upload to GCS
  const gcsUrl = await uploadToGCS(
    thumbPath,
    `thumbnails/${thumbName}`,
    'image/jpeg'
  );

  fs.unlinkSync(thumbPath); // Cleanup temp

  return gcsUrl;
};
