# 🔧 Phase 2 Breakdown: Media Upload & Detection APIs

✅ Key Components

1. API Design

```pl
        POST /upload – Single file

        POST /upload/batch – Multiple files

        GET /media/:id – Check detection result
```

2. File Handling

```plain
        Accept image/video MIME types.

        Enforce file size limits (e.g., max 100MB).

        Use multer (Node.js) or GCP’s signed URLs if you want browser-to-bucket upload later.
```

3. Cloud Storage Integration

```pl
        Store uploads in your GCP Cloud Storage bucket.

        Organize with naming convention: uploads/{uuid}/{filename}.
```

4. Simulated Detection

```pl
        Return mock detection result.

        Later, plug this into your ML model or GCP Cloud Functions/AI Platform.
```

5.  Security

          Auth (basic token-based or Google IAM in future).

          Validate media content-type to block invalid/unsafe uploads.

🧪 Dev/Test Environments

    Use .env.test, .env.dev, etc.

    Create a test GCP project or at least separate test bucket.

    Add a testing script for POST requests using supertest or jest.

🚀 CI/CD (Phase 1 Follow-up)

    Use GitHub Actions to:

        Run pnpm build + tests on push.

        Build Docker image.

        Push to GCR.

        Deploy to Cloud Run (optionally for dev or staging).
