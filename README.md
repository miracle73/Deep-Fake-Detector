# DeepFake Detector (Backend API)

This project is a containerized Node.js (TypeScript) backend service that provides a REST API for detecting deepfake media content. It's deployed on **Google Cloud Run**, exposed via **Google API Gateway**, and can analyze media using an external detection service or model.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Folder Structure](#folder-structure)
- [Setup](#setup)
- [Local Development](#local-development)
- [Dockerization](#dockerization)
- [Deployment Guide](#deployment-guide)
  - [1. GCP Project Setup](#1-gcp-project-setup)
  - [2. Build and Push Docker Image](#2-build-and-push-docker-image)
  - [3. Deploy to Cloud Run](#3-deploy-to-cloud-run)
  - [4. Configure API Gateway](#4-configure-api-gateway)
- [API Usage](#api-usage)
- [License](#license)

---

## Features

- Exposes a `/detect` POST endpoint to analyze a media URL
- Returns deepfake confidence level and boolean result
- Secured via API Gateway (optionally with API key auth)
- Fully containerized

---

## Architecture

```txt
Client
  |
  v
Google API Gateway
  |
  v
Cloud Run Service (Dockerized Node.js API)
  |
  v
Media Detection Logic (Mocked or External Service)
```

---

## Tech Stack

- **Node.js** (with TypeScript)
- **Express.js**
- **Docker**
- **Google Cloud Run**
- **Google Cloud API Gateway**
- **Google Cloud Storage** (optional for media hosting)

---

## Folder Structure

```
.
├── Dockerfile
├── openapi.yaml
├── package.json
├── tsconfig.json
├── src
│   ├── index.ts           # Entry point
│   ├── routes
│   │   └── detect.ts      # /detect route
│   └── services
│       └── deepfakeservice.ts  # Media analysis logic
```

---

## Setup

```bash
# Clone the repo
pnpm install

# Environment setup
cp .env.development.example .env.development
cp .env.production.example .env.production
```

---

## Local Development

```bash
# Start local server
pnpm run dev

# Or build and run
pnpm run build
node dist/index.js
```

---

## Dockerization

Create a production-ready Docker image:

```Dockerfile
# Dockerfile
FROM node:20-alpine
WORKDIR /app
COPY . .
RUN pnpm install && pnpm run build
CMD ["node", "dist/index.js"]
```

Build and tag the Docker image:

```bash
docker build -t deepfake-detector .
```

---

## Deployment Guide

### 1. GCP Project Setup

```bash
gcloud config set project <your-project-id>
gcloud services enable run.googleapis.com
```

### 2. Build and Push Docker Image

```bash
# Tag and push to GCR
docker tag deepfake-detector gcr.io/<your-project-id>/deepfake-detector
docker push gcr.io/<your-project-id>/deepfake-detector
```

### 3. Deploy to Cloud Run

```bash
gcloud run deploy deepfake-detector \
  --image gcr.io/<your-project-id>/deepfake-detector \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --project=<your-project-id>
```

### 4. Configure API Gateway

#### Step 1: Update your `openapi.yaml`

Replace:

- `YOUR_GATEWAY_HOST` with `deepfake-gateway-<hash>.uc.gateway.dev`
- `YOUR_CLOUD_RUN_URL` with the deployed Cloud Run URL (e.g., `https://deepfake-detector-abc123-uc.a.run.app`)

#### Step 2: Create API Config & Gateway

```bash
gcloud api-gateway apis create deepfake-api --project=<your-project-id>

gcloud api-gateway api-configs create deepfake-config \
  --api=deepfake-api \
  --openapi-spec=openapi.yaml \
  --project=<your-project-id>

gcloud api-gateway gateways create deepfake-gateway \
  --api=deepfake-api \
  --api-config=deepfake-config \
  --location=us-central1 \
  --project=<your-project-id>
```

To get your gateway URL:

```bash
gcloud api-gateway gateways describe deepfake-gateway --location=us-central1
```

---

## API Usage

### Endpoint

```http
POST https://<your-gateway-host>/detect
```

### Request Body

```json
{
  "mediaUrl": "https://example.com/video.mp4"
}
```

### Response

```json
{
  "success": true,
  "result": {
    "confidence": "93%",
    "isDeepfake": true,
    "analyzedFrom": "https://example.com/video.mp4"
  }
}
```

### Errors

- `400 Bad Request` – Missing or invalid `mediaUrl`
- `500 Internal Server Error` – Internal detection or server issue

---

## License

Kindly check the license file for details
