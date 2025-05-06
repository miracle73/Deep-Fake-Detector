export type DetectionJobStatus =
  | 'pending'
  | 'processing'
  | 'completed'
  | 'failed';

export interface DetectionResult {
  isDeepfake: boolean;
  confidence: string;
  processedAt?: string;
  analysisTime?: number;
}

export interface DetectionJob {
  id: string;
  status: DetectionJobStatus;
  fileInfo: {
    originalName: string;
    size: number;
    mimetype: string;
    storageUrl: string;
    publicUrl: string;
  };
  result?: DetectionResult;
  error?: string;
  createdAt: string;
  completedAt?: string;
}

// Enhanced job tracking with expiration
export const detectionJobs = new Map<string, DetectionJob>();

// / Enhanced simulation with more realistic timing
export function simulateDetection(jobId: string) {
  const processingTime = Math.floor(2000 + Math.random() * 3000); // 2-5 seconds

  setTimeout(() => {
    const job = detectionJobs.get(jobId);
    if (!job) return;

    const isDeepfake = Math.random() > 0.5;
    const confidence = `${Math.floor(85 + Math.random() * 10)}%`;

    detectionJobs.set(jobId, {
      ...job,
      status: 'completed',
      result: {
        isDeepfake,
        confidence,
        processedAt: new Date().toISOString(),
        analysisTime: processingTime / 1000,
      },
      completedAt: new Date().toISOString(),
    });

    console.log(
      `[Detection Completed] ${jobId} - ${
        isDeepfake ? 'FAKE' : 'REAL'
      } (${confidence})`
    );
  }, processingTime);
}
