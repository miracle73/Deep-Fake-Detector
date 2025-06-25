import logger from '../logger.js';
import { metricClient } from '../../config/gcp.js';

type MetricType = 'detection_count' | 'processing_time' | 'error_count';

interface MetricPayload {
  type: MetricType;
  value: number;
  labels?: Record<string, string>;
  userId?: string;
}

export const pushMetric = async ({
  type,
  value,
  labels = {},
  userId,
}: MetricPayload) => {
  const projectId = await metricClient.getProjectId();
  const metricType = `custom.googleapis.com/safeguard_media/${type}`;

  try {
    const timeSeries = {
      metric: {
        type: metricType,
        labels: {
          project_id: projectId,
          ...(userId && { user_id: userId }),
          ...labels,
        },
      },
      resource: { type: 'global', labels: { project_id: projectId } },
      points: [
        {
          interval: {
            endTime: { seconds: Math.floor(Date.now() / 1000) },
          },
          value: {
            [type === 'processing_time' ? 'doubleValue' : 'int64Value']: value,
          },
        },
      ],
    };

    await metricClient.createTimeSeries({
      name: metricClient.projectPath(projectId),
      timeSeries: [timeSeries],
    });

    logger.debug(`Metric pushed: ${metricType}`, { value, labels });
  } catch (error) {
    logger.error('Failed to push metric', {
      metricType,
      error: error instanceof Error ? error.message : String(error),
    });
  }
};

// await pushMetric({ type: 'detection_count', value: 1 });
// await pushMetric({
//   type: 'processing_time',
//   value: 2.3,
//   labels: { media_type: 'video' }
// });
