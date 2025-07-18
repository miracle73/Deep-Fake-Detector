import torch
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
import json
import time
import hashlib
from typing import List, Dict, Tuple, Union, Optional, Any
import os
import shutil
from datetime import datetime
import matplotlib.pyplot as plt

def ensure_dir(directory: Union[str, Path]) -> Path:
    """Ensure directory exists, create if it doesn't"""
    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path

def get_device() -> torch.device:
    """Get the best available device (GPU if available, else CPU)"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        print("💻 Using CPU")
    
    return device

def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Count model parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params
    }

def format_time(seconds: float) -> str:
    """Format seconds into human-readable time"""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.1f}s"

def get_model_size_mb(model: torch.nn.Module) -> float:
    """Get model size in MB"""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb

def save_config(config: Dict[str, Any], save_path: Union[str, Path]):
    """Save configuration to JSON file"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=2, default=str)

def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        return json.load(f)

def create_experiment_directory(experiment_name: str, base_dir: str = "experiments") -> Path:
    """Create directory for experiment with timestamp"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = Path(base_dir) / f"{experiment_name}_{timestamp}"
    
    # Create subdirectories
    subdirs = ['checkpoints', 'logs', 'results', 'visualizations']
    for subdir in subdirs:
        (exp_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    return exp_dir

def calculate_dataset_statistics(data_dir: Union[str, Path]) -> Dict[str, Any]:
    """Calculate statistics about the dataset"""
    data_dir = Path(data_dir)
    stats = {
        'total_images': 0,
        'splits': {},
        'classes': {},
        'file_sizes': [],
        'image_sizes': []
    }
    
    # Analyze each split
    for split in ['train', 'val', 'test']:
        split_dir = data_dir / 'processed' / split
        if not split_dir.exists():
            continue
        
        split_stats = {'total': 0, 'real': 0, 'fake': 0}
        
        for class_name in ['real', 'fake']:
            class_dir = split_dir / class_name
            if not class_dir.exists():
                continue
            
            # Count images
            image_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
            count = len(image_files)
            split_stats[class_name] = count
            split_stats['total'] += count
            
            # Sample a few images for size analysis
            for img_path in image_files[:10]:  # Sample first 10 images
                try:
                    # File size
                    file_size = img_path.stat().st_size
                    stats['file_sizes'].append(file_size)
                    
                    # Image dimensions
                    with Image.open(img_path) as img:
                        stats['image_sizes'].append(img.size)
                
                except Exception:
                    continue
        
        stats['splits'][split] = split_stats
        stats['total_images'] += split_stats['total']
    
    # Calculate averages
    if stats['file_sizes']:
        stats['avg_file_size_mb'] = np.mean(stats['file_sizes']) / (1024 * 1024)
    
    if stats['image_sizes']:
        widths, heights = zip(*stats['image_sizes'])
        stats['avg_image_size'] = (int(np.mean(widths)), int(np.mean(heights)))
    
    return stats

def visualize_predictions(
    images: List[np.ndarray],
    predictions: List[str],
    confidences: List[float],
    true_labels: Optional[List[str]] = None,
    save_path: Optional[Union[str, Path]] = None,
    max_images: int = 16
) -> plt.Figure:
    """Visualize model predictions"""
    n_images = min(len(images), max_images)
    cols = 4
    rows = (n_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_images):
        row = i // cols
        col = i % cols
        ax = axes[row, col]
        
        # Display image
        ax.imshow(images[i])
        ax.axis('off')
        
        # Create title
        title = f"Pred: {predictions[i]}\nConf: {confidences[i]:.3f}"
        if true_labels:
            title += f"\nTrue: {true_labels[i]}"
            # Color code: green if correct, red if wrong
            color = 'green' if predictions[i] == true_labels[i] else 'red'
            ax.set_title(title, color=color, fontsize=10)
        else:
            ax.set_title(title, fontsize=10)
    
    # Hide empty subplots
    for i in range(n_images, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def create_model_summary(model: torch.nn.Module, input_shape: Tuple[int, ...]) -> str:
    """Create a comprehensive model summary"""
    try:
        from torchsummary import summary
        import io
        import sys
        
        # Capture summary output
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        
        summary(model, input_shape)
        
        sys.stdout = old_stdout
        summary_str = buffer.getvalue()
        
        return summary_str
    
    except ImportError:
        # Fallback if torchsummary not available
        param_info = count_parameters(model)
        size_info = get_model_size_mb(model)
        
        summary_str = f"""
Model Summary:
==============
Total Parameters: {param_info['total_parameters']:,}
Trainable Parameters: {param_info['trainable_parameters']:,}
Non-trainable Parameters: {param_info['non_trainable_parameters']:,}
Model Size: {size_info:.2f} MB
        """
        
        return summary_str

def benchmark_model(
    model: torch.nn.Module,
    input_shape: Tuple[int, ...],
    device: torch.device,
    num_runs: int = 100
) -> Dict[str, float]:
    """Benchmark model inference speed"""
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, *input_shape).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # Benchmark
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time = total_time / num_runs
    fps = 1.0 / avg_time
    
    return {
        'avg_inference_time_ms': avg_time * 1000,
        'fps': fps,
        'total_time_s': total_time,
        'num_runs': num_runs
    }

def clean_old_checkpoints(checkpoint_dir: Union[str, Path], keep_last: int = 5):
    """Clean old checkpoint files, keeping only the most recent ones"""
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        return
    
    # Get all checkpoint files (excluding best_model.pth)
    checkpoint_files = []
    for pattern in ['*.pth', '*.pt']:
        checkpoint_files.extend(checkpoint_dir.glob(pattern))
    
    # Filter out best_model.pth and latest_checkpoint.pth
    checkpoint_files = [
        f for f in checkpoint_files 
        if f.name not in ['best_model.pth', 'latest_checkpoint.pth']
    ]
    
    # Sort by modification time (newest first)
    checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    # Remove old checkpoints
    files_to_remove = checkpoint_files[keep_last:]
    for file_path in files_to_remove:
        try:
            file_path.unlink()
            print(f"🗑️ Removed old checkpoint: {file_path.name}")
        except Exception as e:
            print(f"⚠️ Failed to remove {file_path.name}: {e}")

def calculate_md5(file_path: Union[str, Path]) -> str:
    """Calculate MD5 hash of a file"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def verify_dataset_integrity(data_dir: Union[str, Path]) -> Dict[str, Any]:
    """Verify dataset integrity and detect corrupted images"""
    data_dir = Path(data_dir)
    results = {
        'total_checked': 0,
        'corrupted_files': [],
        'valid_files': 0,
        'errors': []
    }
    
    # Check all image files
    for img_path in data_dir.rglob('*.jpg'):
        results['total_checked'] += 1
        
        try:
            # Try to open and verify image
            with Image.open(img_path) as img:
                img.verify()
            
            # Try to load image data
            with Image.open(img_path) as img:
                img.load()
            
            results['valid_files'] += 1
        
        except Exception as e:
            results['corrupted_files'].append(str(img_path))
            results['errors'].append(f"{img_path}: {str(e)}")
    
    return results

def memory_cleanup():
    """Clean up GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("🧹 GPU memory cache cleared")

def set_random_seeds(seed: int = 42):
    """Set random seeds for reproducibility"""
    import random
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # For reproducible results
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"🎲 Random seeds set to {seed}")
