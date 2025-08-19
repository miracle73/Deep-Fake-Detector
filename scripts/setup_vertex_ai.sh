#!/bin/bash

# setup_vertex_ai.sh
# Automated setup for Vertex AI Video Deepfake Detection Training

set -e  # Exit on any error

echo "🎬 VERTEX AI SETUP FOR VIDEO DEEPFAKE DETECTION"
echo "================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration variables
PROJECT_ID=""
REGION="us-central1"
BUCKET_NAME=""
SERVICE_ACCOUNT_NAME="video-deepfake-trainer"
SERVICE_ACCOUNT_EMAIL=""

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to get user input
get_input() {
    local prompt="$1"
    local var_name="$2"
    local default_value="$3"
    
    if [ -n "$default_value" ]; then
        read -p "$prompt [$default_value]: " input
        eval "$var_name=\${input:-$default_value}"
    else
        read -p "$prompt: " input
        eval "$var_name=\"$input\""
    fi
}

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    if ! command_exists gcloud; then
        print_error "Google Cloud SDK is not installed. Please install it first:"
        print_error "https://cloud.google.com/sdk/docs/install"
        exit 1
    fi
    
    if ! command_exists python3; then
        print_error "Python 3 is not installed"
        exit 1
    fi
    
    print_success "Prerequisites check passed"
}

# Get configuration from user
get_configuration() {
    print_status "Getting configuration..."
    
    # Get current project if available
    CURRENT_PROJECT=$(gcloud config get-value project 2>/dev/null || echo "")
    
    get_input "Enter your Google Cloud Project ID" PROJECT_ID "$CURRENT_PROJECT"
    get_input "Enter region for Vertex AI" REGION "us-central1"
    get_input "Enter bucket name for data storage" BUCKET_NAME "${PROJECT_ID}-video-deepfake-data"
    
    SERVICE_ACCOUNT_EMAIL="${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
    
    echo
    print_status "Configuration:"
    echo "  Project ID: $PROJECT_ID"
    echo "  Region: $REGION"
    echo "  Bucket: $BUCKET_NAME"
    echo "  Service Account: $SERVICE_ACCOUNT_EMAIL"
    echo
    
    read -p "Continue with this configuration? (y/N): " confirm
    if [[ ! $confirm =~ ^[Yy]$ ]]; then
        print_error "Setup cancelled"
        exit 1
    fi
}

# Set project
set_project() {
    print_status "Setting Google Cloud project..."
    
    gcloud config set project "$PROJECT_ID"
    
    if [ $? -eq 0 ]; then
        print_success "Project set to $PROJECT_ID"
    else
        print_error "Failed to set project"
        exit 1
    fi
}

# Enable required APIs
enable_apis() {
    print_status "Enabling required Google Cloud APIs..."
    
    APIS=(
        "aiplatform.googleapis.com"
        "storage.googleapis.com"
        "compute.googleapis.com"
        "containerregistry.googleapis.com"
        "cloudbuild.googleapis.com"
    )
    
    for api in "${APIS[@]}"; do
        print_status "Enabling $api..."
        gcloud services enable "$api" --project="$PROJECT_ID"
        
        if [ $? -eq 0 ]; then
            print_success "Enabled $api"
        else
            print_warning "Failed to enable $api (may already be enabled)"
        fi
    done
}

# Create service account
create_service_account() {
    print_status "Creating service account..."
    
    # Check if service account already exists
    if gcloud iam service-accounts describe "$SERVICE_ACCOUNT_EMAIL" --project="$PROJECT_ID" >/dev/null 2>&1; then
        print_warning "Service account $SERVICE_ACCOUNT_EMAIL already exists"
    else
        gcloud iam service-accounts create "$SERVICE_ACCOUNT_NAME" \
            --display-name="Video Deepfake Trainer" \
            --description="Service account for Vertex AI video deepfake detection training" \
            --project="$PROJECT_ID"
        
        if [ $? -eq 0 ]; then
            print_success "Created service account $SERVICE_ACCOUNT_EMAIL"
        else
            print_error "Failed to create service account"
            exit 1
        fi
    fi
}

# Assign IAM roles
assign_iam_roles() {
    print_status "Assigning IAM roles to service account..."
    
    ROLES=(
        "roles/aiplatform.user"
        "roles/storage.admin"
        "roles/compute.instanceAdmin.v1"
        "roles/logging.logWriter"
        "roles/monitoring.metricWriter"
    )
    
    for role in "${ROLES[@]}"; do
        print_status "Assigning role $role..."
        gcloud projects add-iam-policy-binding "$PROJECT_ID" \
            --member="serviceAccount:$SERVICE_ACCOUNT_EMAIL" \
            --role="$role"
        
        if [ $? -eq 0 ]; then
            print_success "Assigned role $role"
        else
            print_warning "Failed to assign role $role"
        fi
    done
}

# Create service account key
create_service_account_key() {
    print_status "Creating service account key..."
    
    KEY_FILE="service-account-key.json"
    
    if [ -f "$KEY_FILE" ]; then
        print_warning "Service account key file already exists: $KEY_FILE"
        read -p "Overwrite existing key file? (y/N): " overwrite
        if [[ ! $overwrite =~ ^[Yy]$ ]]; then
            print_status "Using existing key file"
            return
        fi
    fi
    
    gcloud iam service-accounts keys create "$KEY_FILE" \
        --iam-account="$SERVICE_ACCOUNT_EMAIL" \
        --project="$PROJECT_ID"
    
    if [ $? -eq 0 ]; then
        print_success "Created service account key: $KEY_FILE"
        print_warning "Keep this file secure and do not commit it to version control!"
        
        # Set environment variable
        export GOOGLE_APPLICATION_CREDENTIALS="$(pwd)/$KEY_FILE"
        echo "export GOOGLE_APPLICATION_CREDENTIALS=\"$(pwd)/$KEY_FILE\"" >> ~/.bashrc
        
    else
        print_error "Failed to create service account key"
        exit 1
    fi
}

# Create storage bucket
create_storage_bucket() {
    print_status "Creating Cloud Storage bucket..."
    
    # Check if bucket already exists
    if gsutil ls -b "gs://$BUCKET_NAME" >/dev/null 2>&1; then
        print_warning "Bucket gs://$BUCKET_NAME already exists"
    else
        gsutil mb -p "$PROJECT_ID" -c STANDARD -l "$REGION" "gs://$BUCKET_NAME"
        
        if [ $? -eq 0 ]; then
            print_success "Created bucket gs://$BUCKET_NAME"
        else
            print_error "Failed to create bucket"
            exit 1
        fi
    fi
    
    # Create directory structure
    print_status "Creating bucket directory structure..."
    
    DIRECTORIES=(
        "datasets/videos/raw"
        "datasets/videos/processed"
        "models/checkpoints"
        "models/experiments"
        "results/training"
        "results/evaluation"
    )
    
    for dir in "${DIRECTORIES[@]}"; do
        echo "" | gsutil cp - "gs://$BUCKET_NAME/$dir/.gitkeep"
    done
    
    print_success "Created bucket directory structure"
}

# Install Python dependencies
install_dependencies() {
    print_status "Installing Python dependencies..."
    
    if [ -f "requirements.txt" ]; then
        pip3 install -r requirements.txt
        print_success "Installed dependencies from requirements.txt"
    else
        print_status "Installing basic dependencies..."
        pip3 install google-cloud-aiplatform google-cloud-storage torch torchvision tqdm pillow
        print_success "Installed basic dependencies"
    fi
}

# Create environment file
create_env_file() {
    print_status "Creating environment configuration file..."
    
    ENV_FILE=".env"
    
    cat > "$ENV_FILE" << EOF
# Google Cloud Configuration
GOOGLE_CLOUD_PROJECT=$PROJECT_ID
GOOGLE_CLOUD_REGION=$REGION
GOOGLE_APPLICATION_CREDENTIALS=$(pwd)/service-account-key.json

# Storage Configuration
GCS_BUCKET_NAME=$BUCKET_NAME
GCS_DATA_PREFIX=datasets/videos
GCS_MODEL_PREFIX=models/experiments

# Vertex AI Configuration
VERTEX_AI_REGION=$REGION
SERVICE_ACCOUNT_EMAIL=$SERVICE_ACCOUNT_EMAIL

# Training Configuration
MACHINE_TYPE=n1-highcpu-16
USE_PREEMPTIBLE=true
MAX_TRAINING_TIME=3600  # 1 hour
EOF

    print_success "Created environment file: $ENV_FILE"
}

# Test setup
test_setup() {
    print_status "Testing setup..."
    
    # Test gcloud authentication
    if gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q "@"; then
        print_success "gcloud authentication: OK"
    else
        print_error "gcloud authentication: FAILED"
        return 1
    fi
    
    # Test bucket access
    if gsutil ls "gs://$BUCKET_NAME" >/dev/null 2>&1; then
        print_success "Storage bucket access: OK"
    else
        print_error "Storage bucket access: FAILED"
        return 1
    fi
    
    # Test Vertex AI API
    if gcloud ai endpoints list --region="$REGION" >/dev/null 2>&1; then
        print_success "Vertex AI API access: OK"
    else
        print_error "Vertex AI API access: FAILED"
        return 1
    fi
    
    print_success "Setup test completed successfully!"
}

# Print next steps
print_next_steps() {
    echo
    print_success "🎉 Vertex AI setup completed successfully!"
    echo
    print_status "Next steps:"
    echo "1. Upload your video dataset:"
    echo "   python scripts/upload_data_to_gcs.py --data-dir /path/to/videos --project-id $PROJECT_ID --bucket-name $BUCKET_NAME"
    echo
    echo "2. Start CPU-optimized training:"
    echo "   python scripts/vertex_ai_training_cpu_optimized.py --bucket-name $BUCKET_NAME"
    echo
    echo "3. Or use the main script:"
    echo "   python main.py --mode upload --data-dir /path/to/videos"
    echo "   python main.py --mode train --use-vertex"
    echo
    print_status "Configuration files created:"
    echo "  - .env (environment variables)"
    echo "  - service-account-key.json (service account credentials)"
    echo
    print_warning "Remember to:"
    echo "  - Keep service-account-key.json secure"
    echo "  - Add .env and *.json to .gitignore"
    echo "  - Set up billing alerts for cost control"
}

# Main execution
main() {
    check_prerequisites
    get_configuration
    set_project
    enable_apis
    create_service_account
    assign_iam_roles
    create_service_account_key
    create_storage_bucket
    install_dependencies
    create_env_file
    
    if test_setup; then
        print_next_steps
    else
        print_error "Setup test failed. Please check the configuration and try again."
        exit 1
    fi
}

# Run main function
main "$@"