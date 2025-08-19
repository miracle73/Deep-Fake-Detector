import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any
from google.cloud import storage

def setup_logging(level: str = 'INFO'):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML or JSON file"""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            return yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            return json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")

def save_model_to_gcs(local_path: str, bucket_name: str, gcs_path: str):
    """Upload model to Google Cloud Storage"""
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(gcs_path)
        
        blob.upload_from_filename(local_path)
        print(f"Model uploaded to gs://{bucket_name}/{gcs_path}")
        
    except Exception as e:
        print(f"Failed to upload model to GCS: {e}")
