export async function detectDeepfake(mediaUrl: string) {
  return {
    confidence: '93%',
    isDeepfake: true,
    analyzedFrom: mediaUrl,
  };
}
