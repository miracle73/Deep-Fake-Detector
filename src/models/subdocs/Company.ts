import { Schema } from 'mongoose';

export const CompanySchema = new Schema(
  {
    name: { type: String, required: true },
    website: { type: String, required: true },
    size: { type: String },
    industry: { type: String },
  },
  { _id: false }
);
