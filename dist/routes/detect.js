import { detectDeepfake } from '../services/deepfakeservice';
export async function detectHandler(req, res) {
    try {
        const { mediaUrl } = req.body;
        const result = await detectDeepfake(mediaUrl);
        res.json({ success: true, result });
    }
    catch (error) {
        res.status(500).json({ success: false, error: error.message });
    }
}
