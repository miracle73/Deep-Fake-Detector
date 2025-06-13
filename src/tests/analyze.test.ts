import path from 'node:path';
import request from 'supertest';

import app from '../index.js';

jest.mock('../services/vertexClient', () => ({
  callVertexAI: jest.fn().mockResolvedValue({
    isDeepfake: true,
    confidence: '97%',
    message: 'Mocked detection success',
  }),
}));

describe('POST /api/analyze', () => {
  it('should return 200 with a valid image upload', async () => {
    const res = await request(app)
      .post('/api/analyze')
      .attach('image', path.join(__dirname, 'mock.jpg'));

    expect(res.status).toBe(200);
    expect(res.body).toHaveProperty('success', true);
    expect(res.body.data.result).toHaveProperty('isDeepfake');
  });

  it('should return 400 when no file is uploaded', async () => {
    const res = await request(app).post('/api/analyze');
    expect(res.status).toBe(400);
    expect(res.body).toHaveProperty('errorCode', 'NO_FILE');
  });
});
