
# package_for_vertex_ai.sh
# Package your code for Vertex AI training

set -e

PROJECT_ID="your-project-id"  # Replace with your project ID
BUCKET_NAME="second-bucket-video-deepfake"
REGION="us-central1"

echo "📦 PACKAGING CODE FOR VERTEX AI"
echo "================================="

# Create package directory
PACKAGE_DIR="vertex_ai_package"
rm -rf $PACKAGE_DIR
mkdir -p $PACKAGE_DIR

echo "📂 Copying source code..."
# Copy source code
cp -r src/ $PACKAGE_DIR/
cp -r scripts/ $PACKAGE_DIR/
cp requirements.txt $PACKAGE_DIR/

# Create setup.py for the package
cat > $PACKAGE_DIR/setup.py << 'EOF'
from setuptools import setup, find_packages

setup(
    name="video-deepfake-detector",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.13.0",
        "torchvision>=0.14.0", 
        "opencv-python-headless",
        "pillow>=10.1.0",
        "numpy>=1.21.0",
        "google-cloud-aiplatform>=1.35.0",
        "google-cloud-storage>=2.10.0",
        "google-auth>=2.20.0",
        "tqdm>=4.64.0",
        "pyyaml>=6.0",
        "python-dotenv>=1.0.0"
    ]
)
EOF

# Create Dockerfile for custom container (optional)
cat > $PACKAGE_DIR/Dockerfile << 'EOF'
FROM gcr.io/cloud-aiplatform/training/pytorch-cpu.1-13:latest

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy source code
COPY src/ src/
COPY scripts/ scripts/

# Set Python path
ENV PYTHONPATH=/app:$PYTHONPATH

ENTRYPOINT ["python", "scripts/vertex_ai_training_cpu_optimized.py"]
EOF

echo "✅ Code packaged in $PACKAGE_DIR/"

# Upload package to GCS
echo "☁️ Uploading package to GCS..."
gsutil -m cp -r $PACKAGE_DIR gs://$BUCKET_NAME/vertex_ai_package/

echo "🎉 Package uploaded to gs://$BUCKET_NAME/vertex_ai_package/"
echo ""
echo "🚀 Next steps:"
echo "1. Update submit_vertex_ai_job.py with your project details"
echo "2. Run: python submit_vertex_ai_job.py --project-id $PROJECT_ID --service-account your-service-account@$PROJECT_ID.iam.gserviceaccount.com"