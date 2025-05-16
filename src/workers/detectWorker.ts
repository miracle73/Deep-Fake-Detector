import { PubSub } from '@google-cloud/pubsub';
import { callVertexAI } from '../services/vertexClient';

const pubsub = new PubSub();
const subscriptionName = 'detect-media-sub';

const messageHandler = async (message: any) => {
  const job = JSON.parse(message.data.toString());

  console.log(`Processing job: ${job.id}`);

  try {
    const prediction = await callVertexAI(job.fileInfo.publicUrl);
    // update job status somewhere (e.g., memory, Firestore, logs)
    console.log('Prediction:', prediction);
  } catch (err) {
    console.error('Detection failed:', err);
  }

  message.ack();
};

pubsub.subscription(subscriptionName).on('message', messageHandler);
