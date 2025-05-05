# Important Commands for Phase 1 Setup

This document captures all critical commands used throughout the Phase 1 setup — including Docker, Google Cloud Platform (GCP), Cloud Run deployment, API Gateway, and Testing.

---

## 🐳 Docker Commands

- **Build Docker Image**

```bash
docker build -t gcr.io/deepfake-detector-455108/deepfake-detector .
```

- **Push Docker Image to Google Container Registry (GCR)**

```bash
docker push gcr.io/deepfake-detector-455108/deepfake-detector
```

- **View Docker Images Locally**

```bash
docker images
```

---

## ☁️ GCP Commands

- **Deploy Container to Cloud Run**

```bash
gcloud run deploy deepfake-detector \
  --image gcr.io/deepfake-detector-455108/deepfake-detector \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --project=deepfake-detector-455108
```

- **Enable Required GCP APIs (when prompted)**  
  _(example: enabling Cloud Run API)_

```bash
# Done automatically when you are prompted:
Do you want to enable these APIs? (Y/n)
# Choose Y
```

- **Check Deployed Cloud Run Services**

```bash
gcloud run services list
```

- **Get Cloud Run URL**

```bash
gcloud run services describe deepfake-detector \
  --region us-central1 \
  --format 'value(status.url)'
```

## 🛠️ IAM Policy Setup (for Public Access)

- **Manually Set IAM Permissions if Deployment Fails**

```bash
gcloud beta run services add-iam-policy-binding deepfake-detector \
  --region us-central1 \
  --member=allUsers \
  --role=roles/run.invoker
```

This allows unauthenticated users (e.g., API Gateway) to invoke your service.

---

## 📡️ Testing Commands

- **Test the POST `/detect` Endpoint via cURL**

```bash
curl -X POST https://deepfake-gateway-8cz9cte8.uc.gateway.dev/detect \
  -H "Content-Type: application/json" \
  -d '{"mediaUrl": "https://example.com/video.mp4"}'


curl -X POST https://deepfake-gateway-8cz9cte8.uc.gateway.dev/detect \
  -H "Content-Type: application/json" \
  -H "x-api-key: AIzaSyBeefwGOz1hgLjVJnBC1iRZhmxdf_8CMtc" \
  -d '{"mediaUrl": "https://example.com/video.mp4"}'

```

---

## 🔍 Debugging Tips

- **Check Cloud Run Deployment Logs**

Use the log link provided after deploying or visit:

```plain
GCP Console > Logging > Logs Explorer > Resource: Cloud Run Revision
```

## 📦 Bonus: Development Tools

- **Start Local Development**

```bash
pnpm run dev
```

(using something like `ts-node-dev` or `nodemon`)

- **Build for Production**

```bash
pnpm run build
```

(compiles TypeScript files to JavaScript)

---

## ⚙️ Basic Project Structure Overview

```bash
/src
  /routes
    detect.ts
  /services
    deepfakeservice.ts
index.ts
Dockerfile
openapi.yaml
```

---

## 📜 Notes

- API Gateway only allows **POST** on `/detect`, no **GET** (important when testing).
- Docker image names must be unique if versioning manually.
- Environment-specific `.env` files were used (e.g., `.env.development`, `.env.production`).
- Google Cloud Run expects your app to bind to `PORT` environment variable.

This contains various commands from the documentation used in the setup of phase 1.

- **To build and push the docker image**

```pl
docker build -t gcr.io/deepfake-detector-455108/deepfake-detector .

docker push gcr.io/deepfake-detector-455108/deepfake-detector
```

- **To redeploy the image**

```pl
gcloud run deploy deepfake-detector \
  --image gcr.io/deepfake-detector-455108/deepfake-detector \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --project=deepfake-detector-455108

```

- **Test POST - /detect endpoint\***

```curl
curl -X POST https://deepfake-gateway-8cz9cte8.uc.gateway.dev/detect \
  -H "Content-Type: application/json" \
  -d '{"mediaUrl": "https://example.com/video.mp4"}'

```

- **Test POST - /upload endpoint**

```curl

curl -X POST https://deepfake-gateway-8cz9cte8.uc.gateway.dev/media/upload \
 -H "Content-Type: multipart/form-data" \
 -F "media=@/path/to/your/video.mp4"

```

- **Test POST - /upload/batch endpoint**

```curl

curl -X POST https://deepfake-gateway-8cz9cte8.uc.gateway.dev/media/upload/batch \
  -H "Content-Type: multipart/form-data" \
  -F "media=@/path/to/video1.mp4" \
  -F "media=@/path/to/video2.mp4"

```

## Setting up Dev and Test Environments

```pl
# Build Docker image as usual
docker build -t gcr.io/deepfake-detector-455108/deepfake-detector-dev .

# Push it
docker push gcr.io/deepfake-detector-455108/deepfake-detector-dev

# Deploy to Cloud Run (dev)
gcloud run deploy deepfake-detector-dev \
  --image gcr.io/deepfake-detector-455108/deepfake-detector-dev \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --project=deepfake-detector-455108

# Same for test
```

### API GATEWAY CONFIGS

```pl

gcloud api-gateway api-configs create deepfake-config-dev \
  --api=deepfake-api-dev \
  --openapi-spec=openapi-dev.yaml \
  --project=deepfake-detector-455108

gcloud api-gateway gateways create deepfake-gateway-dev \
  --api=deepfake-api-dev \
  --api-config=deepfake-config-dev \
  --location=us-central1 \
  --project=deepfake-detector-455108

```

### GCP Storage

- **To find the GCP bucket name**

```pl
gcloud storage buckets list --project=deepfake-detector-455108


# deep-fake-001

```

- **To identify your Cloud Run service's service account**

```pl
#  Run this to get the service account name:
gcloud run services describe deepfake-detector \
  --platform managed \
  --region us-central1 \
  --project=deepfake-detector-455108 \
  --format="value(spec.template.spec.serviceAccountName)"


# 655166731472-compute@developer.gserviceaccount.com

```

- **To grant the service account permission to use the bucket**

```pl
gcloud projects add-iam-policy-binding deepfake-detector-455108 \
  --member="serviceAccount:655166731472-compute@developer.gserviceaccount.com" \
  --role="roles/storage.objectAdmin"


```
