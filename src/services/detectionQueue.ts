type DetectionJob = {
  status: 'pending' | 'completed' | 'failed';
  result?: {
    isDeepfake: boolean;
    confidence: string;
  };
};

export const detectionJobs = new Map<string, DetectionJob>();

export function simulateDetection(jobId: string) {
  setTimeout(() => {
    const isDeepfake = Math.random() > 0.5;
    const confidence = `${Math.floor(85 + Math.random() * 10)}%`;

    detectionJobs.set(jobId, {
      status: 'completed',
      result: { isDeepfake, confidence },
    });

    console.log(`[Detection Completed] ${jobId}`);
  }, 3000); // Simulate 3s delay
}
