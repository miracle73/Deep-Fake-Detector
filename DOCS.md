# 🧠 DeepFake Detection API — Docs

## Details

**Base URL:**

```plain
https://deepfake-gateway-8cz9cte8.uc.gateway.dev
```

**Authentication:**
All requests must include an API key in the header:

```http
x-api-key: YOUR_API_KEY_HERE
```

---

### 🔹 `GET /`

**Description:** Returns a welcome message with API metadata.

**Response Example:**

```json
{
  "message": "Welcome to the image detection API",
  "version": "1.0.0",
  "endpoints": {
    "/detect": "POST - Detect objects in an image",
    "/media/upload": "POST - Upload an image for detection",
    "/media/upload/batch": "POST - Upload multiple images for detection"
  }
}
```

---

### 🔹 `POST /detect`

**Description:** Detect deepfakes from a media URL.

**Request Headers:**

```http
Content-Type: application/json
x-api-key: YOUR_API_KEY_HERE
```

**Request Body:**

```json
{
  "mediaUrl": "https://example.com/video.mp4"
}
```

**Response:**

```json
{
  "result": "likely_fake",
  "confidence": 0.92
}
```

---

### 🔹 `POST /media/upload`

**Description:** Upload a single image/video file for detection.

**Content-Type:** `multipart/form-data`

**Form Field:**

- `media`: (File) The file to upload

**Response Example:**

```json
{
  "message": "Upload successful",
  "mediaId": "abc123",
  "result": "likely_real"
}
```

---

### 🔹 `POST /media/upload/batch`

**Description:** Upload multiple image/video files at once.

**Content-Type:** `multipart/form-data`

**Form Field:**

- `media`: (Array of files) Multiple media files

**Response Example:**

```json
[
  {
    "mediaId": "file1",
    "result": "likely_fake"
  },
  {
    "mediaId": "file2",
    "result": "likely_real"
  }
]
```

---
