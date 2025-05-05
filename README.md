# Deep Fake Detector API

A robust API service for detecting deep fake videos and images using machine learning. This service provides endpoints for analyzing individual media files as well as batch processing capabilities.

## Features

- Single media file analysis
- Batch media processing
- Real-time detection status tracking
- Google Cloud Storage integration
- Support for video formats (MP4, MPEG, QuickTime)
- Asynchronous processing with job tracking

## Tech Stack

- Node.js
- TypeScript
- Express.js
- Google Cloud Storage
- Multer (for file uploads)

## Prerequisites

- Node.js (v14 or higher)
- Google Cloud Platform account with Storage API enabled
- Environment variables configured

## Environment Variables

Create a `.env.development` file in the root directory with the following variables:

```env
PORT=8080
GCS_BUCKET_NAME=deep-fake-001
ENV=development
```

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd deep-fake-detector
```

2. Install dependencies:

```bash
npm install
```

3. Build the project:

```bash
npm run build
```

4. Start the server:

```bash
npm start
```

## API Endpoints

### 1. Health Check

```
GET /health
```

Returns the server status.

### 2. Single Media Analysis

```
POST /api/analyze
```

Analyzes a single video file for deep fake detection.

**Request:**

- Content-Type: multipart/form-data
- Body: `media` (video file)

**Response:**

```json
{
  "success": true,
  "uploadedTo": "https://storage.googleapis.com/bucket-name/file-path",
  "result": {
    "isDeepfake": true,
    "confidence": "94%",
    "message": "Detection complete"
  }
}
```

### 3. Batch Media Analysis

```
POST /api/analyze/batch
```

Processes multiple video files (up to 10) simultaneously.

**Request:**

- Content-Type: multipart/form-data
- Body: `media` (multiple video files)

**Response:**

```json
{
  "success": true,
  "count": 2,
  "uploads": [
    {
      "id": "job-uuid",
      "uploadedTo": "https://storage.googleapis.com/bucket-name/file-path",
      "status": "pending"
    }
  ]
}
```

### 4. Check Analysis Status

```
GET /api/analyze/status/:id
```

Retrieves the status of a detection job.

**Response:**

```json
{
  "id": "job-uuid",
  "status": "completed",
  "result": {
    "isDeepfake": true,
    "confidence": "95%"
  }
}
```

## File Upload Restrictions

- Maximum file size: 20MB
- Supported formats:
  - video/mp4
  - video/mpeg
  - video/quicktime

## Error Handling

The API implements comprehensive error handling with appropriate HTTP status codes and detailed error messages. Common error responses include:

- 400: Bad Request (invalid file type, missing file)
- 404: Not Found (job not found)
- 500: Internal Server Error

## Project Structure

```
src/
├── config/         # Configuration files
├── controllers/    # Route controllers
├── middlewares/    # Express middlewares
├── routes/         # API routes
├── services/       # Business logic
└── utils/          # Utility functions
```

## Development

To run the project in development mode:

```bash
npm run dev
```

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

Please, check the license file for details
