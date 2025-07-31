import Analysis from '../models/Analysis.js';

export async function storeAnalysis({
  user,
  confidence,
  file,
  thumbnailUrl,
}: {
  // biome-ignore lint/suspicious/noExplicitAny: <explanation>
  user: any;
  confidence: number;
  file: Express.Multer.File;
  thumbnailUrl: string;
}) {
  const analysis = await Analysis.create({
    userId: user._id,
    fileName: file.originalname,
    thumbnailUrl,
    uploadDate: Date.now(),
    status: 'authentic',
    confidenceScore: confidence,
  });

  user.analysisHistory.push(analysis._id);
  await user.save();

  return analysis;
}
