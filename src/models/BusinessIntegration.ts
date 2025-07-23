import mongoose from 'mongoose';

const businessIntegrationSchema = new mongoose.Schema(
  {
    firstName: {
      type: String,
      required: true,
      min: 2,
      max: 50,
      trim: true,
    },
    lastName: {
      type: String,
      required: true,
      min: 2,
      max: 50,
      trim: true,
    },
    email: {
      type: String,
      required: true,
      unique: true,
      trim: true,
      lowercase: true,
      validate: {
        validator: (v) => {
          return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(v);
        },
      },
    },
    companyName: {
      type: String,
      required: true,
      min: 2,
      max: 100,
      trim: true,
    },
    companyWebsite: {
      type: String,
      required: true,
      validate: {
        validator: (v) => {
          return /^(https?:\/\/)?([\w.-]+)+(:\d+)?(\/[\w.-]*)*\/?$/.test(v);
        },
      },
      trim: true,
    },
    businessNeeds: {
      type: String,
      required: true,
      min: 10,
      max: 500,
      trim: true,
    },
    status: {
      type: String,
      enum: ['new', 'contacted', 'closed'],
      default: 'new',
    },
  },
  {
    timestamps: true,
  }
);

const BusinessIntegration = mongoose.model(
  'BusinessIntegration',
  businessIntegrationSchema
);
export default BusinessIntegration;
