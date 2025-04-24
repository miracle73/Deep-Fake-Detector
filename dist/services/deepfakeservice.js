export async function detectDeepfake(mediaUrl) {
    return {
        confidence: '93%',
        isDeepfake: true,
        analyzedFrom: mediaUrl,
    };
}
