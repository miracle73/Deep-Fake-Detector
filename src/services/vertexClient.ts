import { PredictionServiceClient } from '@google-cloud/aiplatform';

const client = new PredictionServiceClient();

const project = process.env.PROJECT_NAME;
const location = process.env.PROJECT_LOCATION;
const endpointId = process.env.MODEL_ENDPOINT_ID;

const endpoint = `projects/${project}/locations/${location}/endpoints/${endpointId}`;

export const callVertexAI = async (mediaUrl: string) => {
  const request = {
    endpoint,
    instances: [{ image_url: mediaUrl }],
    parameters: {},
  };

  const [response] = await client.predict(request);

  if (!response.predictions || response.predictions.length === 0) {
    throw new Error('No predictions found');
  }
  if (!response.predictions[0]) {
    throw new Error('No predictions found in the response');
  }

  const predictions = response.predictions[0];
  return predictions;
};

export const callVertexAIBatch = async (mediaUrls: string[]) => {
  const request = {
    endpoint,
    instances: mediaUrls.map((url) => ({ image_url: url })),
    parameters: {},
  };

  const [response] = await client.predict(request);

  if (!response.predictions || response.predictions.length === 0) {
    throw new Error('No predictions found');
  }

  if (!response.predictions[0]) {
    throw new Error('No predictions found in the response');
  }

  const predictions = response.predictions[0] || [];

  return predictions;
};
