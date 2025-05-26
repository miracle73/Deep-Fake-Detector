import { PredictionServiceClient } from '@google-cloud/aiplatform';
import type { protos } from '@google-cloud/aiplatform';
type IValue = protos.google.cloud.aiplatform.v1.IValue;

const client = new PredictionServiceClient();

const project = process.env.PROJECT_NAME;
const location = process.env.PROJECT_LOCATION;
const endpointId = process.env.MODEL_ENDPOINT_ID;

const endpoint = `projects/${project}/locations/${location}/endpoints/${endpointId}`;

interface PredictionResult {
  isDeepfake: boolean;
  confidence: number;
  message: string;
  jobId?: string;
}

export const callVertexAI = async (
  mediaUrl: string
): Promise<PredictionResult> => {
  const request = {
    endpoint,
    instances: [{ stringValue: mediaUrl }] as IValue[],
    parameters: {},
  };

  const [response] = await client.predict(request);

  if (!response.predictions || response.predictions.length === 0) {
    throw new Error('No predictions found');
  }

  const prediction = response.predictions[0];
  if (!prediction?.structValue?.fields) {
    throw new Error('Invalid prediction format');
  }

  const fields = prediction.structValue.fields;
  return {
    isDeepfake: fields.isDeepfake?.boolValue ?? false,
    confidence: fields.confidence?.numberValue ?? 0,
    message: fields.message?.stringValue ?? 'No message',
    jobId: fields.jobId?.stringValue ?? undefined,
  };
};

export const callVertexAIBatch = async (
  mediaUrls: string[]
): Promise<PredictionResult[]> => {
  const request = {
    endpoint,
    instances: mediaUrls.map((url) => ({ stringValue: url })) as IValue[],
    parameters: {},
  };

  const [response] = await client.predict(request);

  if (!response.predictions || response.predictions.length === 0) {
    throw new Error('No predictions found');
  }

  return response.predictions.map((prediction) => {
    if (!prediction?.structValue?.fields) {
      throw new Error('Invalid prediction format');
    }

    const fields = prediction.structValue.fields;
    return {
      isDeepfake: fields.isDeepfake?.boolValue ?? false,
      confidence: fields.confidence?.numberValue ?? 0,
      message: fields.message?.stringValue ?? 'No message',
      jobId: fields.jobId?.stringValue ?? undefined,
    };
  });
};
