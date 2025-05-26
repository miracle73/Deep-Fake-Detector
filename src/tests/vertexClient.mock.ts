export const callVertexAI = jest.fn().mockResolvedValue({
  isDeepfake: true,
  confidence: '98%',
  message: 'Mocked detection',
});
