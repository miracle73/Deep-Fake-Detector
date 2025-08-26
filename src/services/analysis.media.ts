import Analysis from '../models/Analysis.js';

export async function storeAnalysis({
  user,
  confidence,
  file,
  thumbnailUrl,
  result,
}: {
  // biome-ignore lint/suspicious/noExplicitAny: <explanation>
  user: any;
  confidence: number;
  file: Express.Multer.File;
  thumbnailUrl: string;
  result: { predicted_class: string; is_deepfake: boolean };
}) {
  const analysis = await Analysis.create({
    userId: user._id,
    fileName: file.originalname,
    thumbnailUrl,
    uploadDate: Date.now(),
    predictedClass: result.predicted_class,
    isDeepfake: result.is_deepfake,
    confidenceScore: confidence,
  });

  console.log('this is the analysis', analysis);

  user.analysisHistory.push(analysis._id);
  await user.save();

  return analysis;
}
