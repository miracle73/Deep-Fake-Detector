import axios from 'axios';
import * as fs from 'fs';
import * as path from 'node:path';
import { v4 as uuidv4 } from 'uuid';

export interface DownloadedFile {
  buffer: Buffer;
  originalName: string;
  mimeType: string;
  size: number;
}

export class MediaDownloader {
  public async callUs() {
    console.log('hello world');
  }

  static async downloadFromUrl(url: string): Promise<DownloadedFile> {
    try {
      const response = await axios({
        method: 'GET',
        url: url,
        responseType: 'arraybuffer',
        timeout: 30000, // 30 seconds timeout
        maxContentLength: 50 * 1024 * 1024, // 50MB max size
      });

      const originalName = this.extractFilename(url) || `media_${uuidv4()}`;

      // Get content type from response headers
      const mimeType =
        response.headers['content-type'] || 'application/octet-stream';

      return {
        buffer: Buffer.from(response.data),
        originalName,
        mimeType,
        size: response.data.length,
      };
    } catch (error) {
      throw new Error(`Failed to download media from URL: ${error}`);
    }
  }

  private static extractFilename(url: string): string {
    try {
      const urlObj = new URL(url);
      const pathname = urlObj.pathname;
      return path.basename(pathname);
    } catch {
      return 'null_filename';
    }
  }

  static isValidUrl(url: string): boolean {
    try {
      new URL(url);
      return true;
    } catch {
      return false;
    }
  }

  static isAllowedFileType(mimeType: string): boolean {
    const allowedTypes = [
      'image/jpeg',
      'image/png',
      'image/gif',
      'image/webp',
      'video/mp4',
      'video/mpeg',
      'video/quicktime',
      'audio/mpeg',
      'audio/wav',
    ];
    return allowedTypes.includes(mimeType);
  }
}
