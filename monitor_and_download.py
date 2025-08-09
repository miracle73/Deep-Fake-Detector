#!/usr/bin/env python3
"""
monitor_and_download.py - Monitor Vertex AI training and download results
"""

import os
import sys
import time
import json
from pathlib import Path
from google.cloud import aiplatform
from google.cloud import storage
from datetime import datetime
import argparse

class TrainingMonitor:
    def __init__(self, project_id: str, bucket_name: str, region: str):
        """Initialize monitor"""
        self.project_id = project_id
        self.bucket_name = bucket_name
        self.region = region
        
        # Initialize clients
        aiplatform.init(project=project_id, location=region)
        self.storage_client = storage.Client(project=project_id)
        self.bucket = self.storage_client.bucket(bucket_name)
    
    def list_training_jobs(self, limit: int = 10):
        """List recent training jobs"""
        print("📋 Recent Training Jobs:")
        print("=" * 60)
        
        jobs = aiplatform.CustomJob.list(
            filter=f'display_name:"deepfake_detector"',
            order_by='create_time desc',
            project=self.project_id,
            location=self.region
        )
        
        for i, job in enumerate(jobs[:limit]):
            print(f"\n{i+1}. {job.display_name}")
            print(f"   State: {job.state}")
            print(f"   Created: {job.create_time}")
            print(f"   Resource: {job.resource_name}")
            
            if job.state == "JOB_STATE_SUCCEEDED":
                print(f"   ✅ Completed successfully")
            elif job.state == "JOB_STATE_FAILED":
                print(f"   ❌ Failed")
            elif job.state == "JOB_STATE_RUNNING":
                print(f"   🏃 Currently running")
        
        return jobs[:limit]
    
    def monitor_job(self, job_name: str, refresh_interval: int = 30):
        """Monitor a specific training job"""
        print(f"📊 Monitoring job: {job_name}")
        print("Press Ctrl+C to stop monitoring")
        print("=" * 60)
        
        try:
            job = aiplatform.CustomJob.get(job_name)
            
            while job.state in ["JOB_STATE_PENDING", "JOB_STATE_RUNNING"]:
                # Clear screen (optional)
                os.system('clear' if os.name == 'posix' else 'cls')
                
                print(f"Job: {job.display_name}")
                print(f"State: {job.state}")
                print(f"Started: {job.start_time if job.start_time else 'Not started'}")
                print(f"Duration: {self._get_duration(job)}")
                
                # Check for logs
                if job.state == "JOB_STATE_RUNNING":
                    print("\n📝 Recent Logs:")
                    self._print_recent_logs(job_name)
                
                # Check for metrics
                self._check_metrics(job_name)
                
                if job.state == "JOB_STATE_SUCCEEDED":
                    print("\n✅ Training completed successfully!")
                    break
                elif job.state == "JOB_STATE_FAILED":
                    print("\n❌ Training failed!")
                    print(f"Error: {job.error}")
                    break
                
                print(f"\n🔄 Refreshing in {refresh_interval} seconds...")
                time.sleep(refresh_interval)
                
                # Refresh job status
                job = aiplatform.CustomJob.get(job_name)
        
        except KeyboardInterrupt:
            print("\n⏹️ Monitoring stopped by user")
        except Exception as e:
            print(f"❌ Error monitoring job: {e}")
    
    def _get_duration(self, job):
        """Calculate job duration"""
        if not job.start_time:
            return "Not started"
        
        end_time = job.end_time if job.end_time else datetime.now(job.start_time.tzinfo)
        duration = end_time - job.start_time
        
        hours = int(duration.total_seconds() // 3600)
        minutes = int((duration.total_seconds() % 3600) // 60)
        
        return f"{hours}h {minutes}m"
    
    def _print_recent_logs(self, job_name: str):
        """Print recent logs from Cloud Logging"""
        try:
            from google.cloud import logging
            
            logging_client = logging.Client(project=self.project_id)
            
            # Get logs for this job
            filter_str = f'resource.labels.job_id="{job_name.split("/")[-1]}"'
            
            entries = logging_client.list_entries(
                filter_=filter_str,
                order_by=logging.DESCENDING,
                max_results=10
            )
            
            for entry in entries:
                print(f"  {entry.timestamp}: {entry.payload}")
        
        except Exception as e:
            print(f"  Could not fetch logs: {e}")
    
    def _check_metrics(self, job_name: str):
        """Check training metrics from GCS"""
        try:
            # Look for metrics file in GCS
            job_id = job_name.split("/")[-1]
            metrics_prefix = f"logs/{job_id}/metrics"
            
            blobs = self.bucket.list_blobs(prefix=metrics_prefix, max_results=1)
            
            for blob in blobs:
                # Download and display latest metrics
                metrics_json = blob.download_as_text()
                metrics = json.loads(metrics_json)
                
                print("\n📈 Latest Metrics:")
                if 'epoch' in metrics:
                    print(f"  Epoch: {metrics['epoch']}")
                if 'train_acc' in metrics:
                    print(f"  Train Accuracy: {metrics['train_acc']:.4f}")
                if 'val_acc' in metrics:
                    print(f"  Val Accuracy: {metrics['val_acc']:.4f}")
                if 'loss' in metrics:
                    print(f"  Loss: {metrics['loss']:.4f}")
        
        except Exception:
            pass  # Metrics might not be available yet
    
    def download_model(self, job_name: str, output_dir: str = "downloaded_models"):
        """Download trained model from GCS"""
        print(f"📥 Downloading model from job: {job_name}")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Extract job ID
        job_id = job_name.split("/")[-1] if "/" in job_name else job_name
        
        # Look for model files
        model_prefix = f"models/{job_id}/"
        
        print(f"🔍 Looking for models in gs://{self.bucket_name}/{model_prefix}")
        
        blobs = self.bucket.list_blobs(prefix=model_prefix)
        downloaded_files = []
        
        for blob in blobs:
            if blob.name.endswith('.pth') or blob.name.endswith('.pt'):
                # Download file
                filename = Path(blob.name).name
                local_path = output_path / filename
                
                print(f"  Downloading {filename}...")
                blob.download_to_filename(str(local_path))
                downloaded_files.append(local_path)
                
                # Get file size
                file_size_mb = local_path.stat().st_size / (1024 * 1024)
                print(f"  ✅ Downloaded {filename} ({file_size_mb:.2f} MB)")
        
        if downloaded_files:
            print(f"\n✅ Downloaded {len(downloaded_files)} model files to {output_path}")
            
            # Download metrics if available
            self._download_metrics(job_id, output_path)
            
            # Download config if available
            self._download_config(job_id, output_path)
        else:
            print(f"❌ No model files found for job {job_id}")
        
        return downloaded_files
    
    def _download_metrics(self, job_id: str, output_path: Path):
        """Download training metrics"""
        try:
            metrics_blob = self.bucket.blob(f"results/{job_id}/metrics.json")
            if metrics_blob.exists():
                local_metrics_path = output_path / "metrics.json"
                metrics_blob.download_to_filename(str(local_metrics_path))
                print(f"  ✅ Downloaded metrics.json")
                
                # Display final metrics
                with open(local_metrics_path, 'r') as f:
                    metrics = json.load(f)
                    
                    print("\n📊 Final Training Metrics:")
                    for key, value in metrics.items():
                        if isinstance(value, (int, float)):
                            print(f"    {key}: {value:.4f}")
        except Exception:
            pass
    
    def _download_config(self, job_id: str, output_path: Path):
        """Download training configuration"""
        try:
            config_blob = self.bucket.blob(f"models/{job_id}/config.yaml")
            if config_blob.exists():
                local_config_path = output_path / "config.yaml"
                config_blob.download_to_filename(str(local_config_path))
                print(f"  ✅ Downloaded config.yaml")
        except Exception:
            pass
    
    def download_all_results(self, job_name: str, output_dir: str = "results"):
        """Download all results from a training job"""
        print(f"📦 Downloading all results from job: {job_name}")
        
        output_path = Path(output_dir)
        job_id = job_name.split("/")[-1] if "/" in job_name else job_name
        job_output_path = output_path / job_id
        job_output_path.mkdir(parents=True, exist_ok=True)
        
        # Download models
        models_path = job_output_path / "models"
        models_path.mkdir(exist_ok=True)
        self.download_model(job_name, str(models_path))
        
        # Download logs
        logs_path = job_output_path / "logs"
        logs_path.mkdir(exist_ok=True)
        self._download_logs(job_id, logs_path)
        
        # Download visualizations if any
        viz_path = job_output_path / "visualizations"
        viz_path.mkdir(exist_ok=True)
        self._download_visualizations(job_id, viz_path)
        
        print(f"\n✅ All results downloaded to {job_output_path}")
        
        # Create summary file
        self._create_summary(job_name, job_output_path)
    
    def _download_logs(self, job_id: str, output_path: Path):
        """Download training logs"""
        try:
            log_prefix = f"logs/{job_id}/"
            blobs = self.bucket.list_blobs(prefix=log_prefix)
            
            for blob in blobs:
                if blob.name.endswith('.log') or blob.name.endswith('.txt'):
                    filename = Path(blob.name).name
                    local_path = output_path / filename
                    blob.download_to_filename(str(local_path))
                    print(f"  ✅ Downloaded {filename}")
        except Exception as e:
            print(f"  ⚠️ Could not download logs: {e}")
    
    def _download_visualizations(self, job_id: str, output_path: Path):
        """Download any visualization files"""
        try:
            viz_prefix = f"results/{job_id}/visualizations/"
            blobs = self.bucket.list_blobs(prefix=viz_prefix)
            
            for blob in blobs:
                if blob.name.endswith(('.png', '.jpg', '.pdf', '.html')):
                    filename = Path(blob.name).name
                    local_path = output_path / filename
                    blob.download_to_filename(str(local_path))
                    print(f"  ✅ Downloaded {filename}")
        except Exception:
            pass
    
    def _create_summary(self, job_name: str, output_path: Path):
        """Create a summary file for the training job"""
        try:
            job = aiplatform.CustomJob.get(job_name)
            
            summary = {
                'job_name': job.display_name,
                'job_id': job_name.split("/")[-1],
                'state': job.state,
                'created': str(job.create_time),
                'started': str(job.start_time) if job.start_time else None,
                'ended': str(job.end_time) if job.end_time else None,
                'duration': self._get_duration(job),
                'machine_spec': job.job_spec.worker_pool_specs[0].machine_spec.__dict__,
                'dataset_info': {
                    'total_images': 297368,
                    'real_images': 155724,
                    'fake_images': 141644
                }
            }
            
            summary_path = output_path / "training_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            print(f"  ✅ Created training_summary.json")
        
        except Exception as e:
            print(f"  ⚠️ Could not create summary: {e}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Monitor and download Vertex AI training results')
    parser.add_argument('--action', choices=['list', 'monitor', 'download'], 
                       required=True, help='Action to perform')
    parser.add_argument('--job-name', help='Job name or resource name')
    parser.add_argument('--output-dir', default='downloaded_models', 
                       help='Output directory for downloads')
    parser.add_argument('--refresh', type=int, default=30, 
                       help='Refresh interval for monitoring (seconds)')
    
    args = parser.parse_args()
    
    # Load environment
    from dotenv import load_dotenv
    load_dotenv()
    
    project_id = os.getenv('PROJECT_ID')
    bucket_name = os.getenv('BUCKET_NAME')
    region = os.getenv('REGION', 'us-central1')
    
    if not project_id or not bucket_name:
        print("❌ Please set PROJECT_ID and BUCKET_NAME in .env")
        sys.exit(1)
    
    # Initialize monitor
    monitor = TrainingMonitor(project_id, bucket_name, region)
    
    if args.action == 'list':
        monitor.list_training_jobs()
    
    elif args.action == 'monitor':
        if not args.job_name:
            print("❌ Please provide --job-name for monitoring")
            sys.exit(1)
        monitor.monitor_job(args.job_name, args.refresh)
    
    elif args.action == 'download':
        if not args.job_name:
            print("❌ Please provide --job-name for download")
            sys.exit(1)
        monitor.download_all_results(args.job_name, args.output_dir)

if __name__ == "__main__":
    main()