## ✅ Phase 3 – Real Detection Integration with Vertex AI

### 🔹 1. 🔐 Set Up Secure Access

- [ ] Ensure your Cloud Run service (or local dev) has access to call Vertex AI.
- [ ] If running locally:

  - Set `GOOGLE_APPLICATION_CREDENTIALS` to your service account key JSON:

    ```bash
    export GOOGLE_APPLICATION_CREDENTIALS=./your-key.json
    ```

- [ ] Make sure the service account has `roles/aiplatform.user`.

---

### 🔹 2. 🔍 Locate the Model Endpoint

- [ ] Ask your teammate for:

  - **Endpoint ID**
  - **Project ID**
  - **Region** where it’s deployed

You can also list them with:

```bash
gcloud ai endpoints list --region=us-central1
```

---

### 🔹 3. 🧠 Write the Model Caller Function

Use the SDK to create a helper like this:

```ts
import { PredictionServiceClient } from '@google-cloud/aiplatform';

const client = new PredictionServiceClient();

export const callVertexAI = async (mediaUrl: string) => {
  const project = 'your-project-id';
  const location = 'us-central1';
  const endpointId = 'your-endpoint-id';

  const endpoint = `projects/${project}/locations/${location}/endpoints/${endpointId}`;

  const request = {
    endpoint,
    instances: [{ image_url: mediaUrl }],
    parameters: {},
  };

  const [response] = await client.predict(request);
  return response.predictions[0];
};
```

---

### 🔹 4. 🔁 Replace Mock Detection

- [ ] Go to `analyze` and `analyzeBulkMedia` handlers
- [ ] After uploading to GCS, call `callVertexAI()` with the public URL
- [ ] Format the result and return it

---

### 🔹 5. 🧪 Test Everything

- [ ] Upload valid media
- [ ] Check Vertex AI is receiving input
- [ ] Confirm output is returned and parsed correctly
- [ ] Handle failures (e.g., bad input, timeouts)

---

### 🔹 6. 📝 Update API Docs

- [ ] Reflect the real detection output shape
- [ ] Update example responses
- [ ] Add error scenarios (e.g., model timeout, prediction failure)

---

### 🔹 7. 🧠 Optional Enhancements

Later in this phase or next:

- [ ] Queue detection via **Pub/Sub**
- [ ] Store detection results in **Firestore** or **BigQuery**
- [ ] Add metrics/logging for detection success/failures

---
