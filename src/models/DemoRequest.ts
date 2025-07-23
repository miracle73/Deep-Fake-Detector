import mongoose from 'mongoose';

const demoRequestSchema = new mongoose.Schema(
  {
    firstName: {
      type: String,
      required: true,
      trim: true,
      min: 3,
      max: 50,
    },
    lastName: {
      type: String,
      required: true,
      trim: true,
      min: 3,
      max: 50,
    },
    email: {
      type: String,
      lowercase: true,
      trim: true,
      required: true,
      unique: true,
    },
    role: {
      type: String,
      enum: [
        'Content Creator/Influencer',
        'Journalist/Reporter',
        'Educator/Teacher',
        'Researcher/Academic',
        'Freelancer/Consultant',
        'Student',
        'Individual User',
        'Other',
      ],
      default: 'Other',
      required: true,
    },
    goal: {
      type: String,
      required: true,
      enum: [
        'Verify content i receive',
        'Protect my personal brand',
        'Fact-checking and research',
        'Teaching/learning about deepfakes',
        'Detect manipulated media',
        'General digital security',
        'Just curious about the technology',
      ],
      default: 'Detect manipulated media',
    },
    contentType: {
      type: String,
      enum: [
        'Videos',
        'Images/Photos',
        'Audio content',
        'Social media posts',
        'News articles',
        'User-generated content',
        'Mixed content types',
      ],
    },
    urgencyLevel: {
      type: String,
      enum: [
        'Need help right now',
        'Within this week',
        'Within this month',
        'Planning for the future',
        'Just exploring options',
      ],
    },
    metadata: {
      type: String,
      default: '',
    },
    demoMailSent: {
      type: Boolean,
      default: false,
    },
  },
  {
    timestamps: true,
  }
);

const DemoRequest = mongoose.model('DemoRequest', demoRequestSchema);
export default DemoRequest;
