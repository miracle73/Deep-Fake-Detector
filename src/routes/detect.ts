import { Request, Response } from 'express';
import { detectDeepfake } from '../services/deepfakeservice.js';

export async function detectHandler(req: Request, res: Response) {
  try {
    const { mediaUrl } = req.body;
    const result = await detectDeepfake(mediaUrl);
    res.json({ success: true, result });
  } catch (error: any) {
    res.status(500).json({ success: false, error: error.message });
  }
}
