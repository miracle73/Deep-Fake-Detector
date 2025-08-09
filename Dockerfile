
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    timm==0.9.12 \
    google-cloud-storage==2.10.0 \
    google-cloud-aiplatform==1.38.0 \
    pillow==10.1.0 \
    tqdm==4.66.1 \
    numpy==1.24.4 \
    torchmetrics==1.2.0

# Copy training script
COPY vertex_training_script.py /app/train.py
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV BUCKET_NAME=first-bucket-deep-fake-detector
ENV JOB_NAME=deepfake-detector-20250809_175159

# Run training
CMD ["python", "train.py"]
