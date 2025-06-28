import mongoose, { Schema } from 'mongoose';

const AnalysisSchema = new mongoose.Schema({
  userId: {
    type: Schema.Types.ObjectId,
    ref: 'User',
    required: true,
  },
  fileName: {
    type: String,
    required: true,
    trim: true,
  },
  thumbnailUrl: {
    type: String,
    required: true,
    trim: true,
  },
  uploadDate: {
    type: Date,
    required: true,
    default: new Date(),
  },
  status: {
    type: String,
    enum: ['authentic', 'uncertain', 'deepfake'],
    required: true,
    default: 'uncertain',
  },
  confidenceScore: {
    type: Number,
    required: true,
    min: 0,
    max: 100,
    default: 0,
  },
});

const Analysis = mongoose.model('Analysis', AnalysisSchema);

export default Analysis;
