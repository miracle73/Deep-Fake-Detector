#!/bin/bash
# setup_vertex_ai.sh - Setup script for Vertex AI in GitHub Codespaces

echo "🚀 Setting up Google Cloud and Vertex AI for Deepfake Detection Training"
echo "========================================================================="

# Step 1: Install Google Cloud SDK
echo "📦 Installing Google Cloud SDK..."
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
sudo apt-get update && sudo apt-get install -y google-cloud-sdk

# Step 2: Set up environment variables
echo "🔧 Setting up environment variables..."
read -p "Enter your GCP Project ID: " PROJECT_ID
read -p "Enter your GCS bucket name (for data/models): " BUCKET_NAME
read -p "Enter your preferred region (e.g., us-central1): " REGION

export PROJECT_ID=$PROJECT_ID
export BUCKET_NAME=$BUCKET_NAME
export REGION=$REGION
export VERTEX_AI_LOCATION=$REGION

# Save to .env file
cat > .env << EOF
PROJECT_ID=$PROJECT_ID
BUCKET_NAME=$BUCKET_NAME
REGION=$REGION
VERTEX_AI_LOCATION=$REGION
EOF

echo "✅ Environment variables saved to .env"

# Step 3: Authenticate with GCP
echo "🔐 Authenticating with Google Cloud..."
gcloud auth login
gcloud config set project $PROJECT_ID

# Step 4: Enable required APIs
echo "🔌 Enabling required Google Cloud APIs..."
gcloud services enable compute.googleapis.com
gcloud services enable storage-api.googleapis.com
gcloud services enable aiplatform.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable artifactregistry.googleapis.com

# Step 5: Create GCS bucket if it doesn't exist
echo "🪣 Creating GCS bucket..."
gsutil mb -l $REGION gs://$BUCKET_NAME 2>/dev/null || echo "Bucket already exists"

# Step 6: Install Python dependencies for Vertex AI
echo "🐍 Installing Vertex AI Python SDK..."
pip install google-cloud-aiplatform google-cloud-storage

echo "✅ Setup complete! You're ready to train on Vertex AI"