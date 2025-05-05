export async function mockDetect(fileUrl: string) {
  return {
    confidence: `${Math.floor(80 + Math.random() * 20)}%`,
    isDeepfake: Math.random() > 0.5,
    analyzedFrom: fileUrl,
  };
}
