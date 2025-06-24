import logger from '../logger.js';
import { metricClient } from '../../config/gcp.js';

export const pushDetectionMetric = async () => {
  const projectId = await metricClient.getProjectId();

  const dataPoint = {
    interval: { endTime: { seconds: Date.now() / 1000 } },
    value: { int64Value: 1 },
  };

  const timeSeries = {
    metric: {
      type: 'custom.googleapis.com/safeguard_media/detection_count',
      labels: { project_id: projectId },
    },
    resource: {
      type: 'global',
      labels: { project_id: projectId },
    },
    points: [dataPoint],
  };

  await metricClient.createTimeSeries({
    name: metricClient.projectPath(projectId),
    timeSeries: [timeSeries],
  });

  logger.info('Detection metric pushed successfully');
};
