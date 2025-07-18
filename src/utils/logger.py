import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Union
import json
import os

class DeepfakeLogger:
    """Professional logging system for deepfake detection project"""
    
    def __init__(
        self,
        name: str = "deepfake_detector",
        log_dir: str = "logs",
        log_level: str = "INFO",
        enable_file_logging: bool = True,
        enable_console_logging: bool = True,
        max_file_size_mb: int = 100,
        backup_count: int = 5
    ):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_level = getattr(logging, log_level.upper())
        self.enable_file_logging = enable_file_logging
        self.enable_console_logging = enable_console_logging
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        self.backup_count = backup_count
        
        # Create log directory
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup logger
        self.logger = self._setup_logger()
        
        # Track metrics and events
        self.metrics_log = []
        self.events_log = []
    
    def _setup_logger(self) -> logging.Logger:
        """Setup the logger with appropriate handlers"""
        logger = logging.getLogger(self.name)
        logger.setLevel(self.log_level)
        
        # Clear any existing handlers
        logger.handlers = []
        
        # Create formatter
        formatter = logging.Formatter(
            fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        if self.enable_console_logging:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.log_level)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # File handler with rotation
        if self.enable_file_logging:
            from logging.handlers import RotatingFileHandler
            
            log_file = self.log_dir / f"{self.name}.log"
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=self.max_file_size_bytes,
                backupCount=self.backup_count
            )
            file_handler.setLevel(self.log_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        self.logger.info(message, extra=kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self.logger.debug(message, extra=kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self.logger.warning(message, extra=kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message"""
        self.logger.error(message, extra=kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message"""
        self.logger.critical(message, extra=kwargs)
    
    def log_training_start(self, config: dict):
        """Log training start with configuration"""
        self.info("🚀 Training Started")
        self.info(f"Configuration: {json.dumps(config, indent=2, default=str)}")
        
        event = {
            'timestamp': datetime.now().isoformat(),
            'event': 'training_start',
            'config': config
        }
        self.events_log.append(event)
    
    def log_epoch_metrics(self, epoch: int, train_metrics: dict, val_metrics: dict):
        """Log epoch metrics"""
        self.info(f"Epoch {epoch} - Train: {self._format_metrics(train_metrics)}")
        self.info(f"Epoch {epoch} - Val: {self._format_metrics(val_metrics)}")
        
        metrics_entry = {
            'timestamp': datetime.now().isoformat(),
            'epoch': epoch,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics
        }
        self.metrics_log.append(metrics_entry)
    
    def log_model_checkpoint(self, epoch: int, checkpoint_path: str, is_best: bool = False):
        """Log model checkpoint save"""
        status = "BEST" if is_best else "REGULAR"
        self.info(f"💾 {status} checkpoint saved at epoch {epoch}: {checkpoint_path}")
        
        event = {
            'timestamp': datetime.now().isoformat(),
            'event': 'checkpoint_save',
            'epoch': epoch,
            'path': checkpoint_path,
            'is_best': is_best
        }
        self.events_log.append(event)
    
    def log_evaluation_results(self, dataset: str, metrics: dict):
        """Log evaluation results"""
        self.info(f"📊 Evaluation on {dataset}: {self._format_metrics(metrics)}")
        
        event = {
            'timestamp': datetime.now().isoformat(),
            'event': 'evaluation',
            'dataset': dataset,
            'metrics': metrics
        }
        self.events_log.append(event)
    
    def log_inference(self, image_path: str, prediction: dict):
        """Log inference result"""
        self.info(f"🔍 Inference on {image_path}: {prediction['predicted_class']} "
                 f"(confidence: {prediction['confidence']:.4f})")
        
        event = {
            'timestamp': datetime.now().isoformat(),
            'event': 'inference',
            'image_path': image_path,
            'prediction': prediction
        }
        self.events_log.append(event)
    
    def log_error_with_traceback(self, error: Exception, context: str = ""):
        """Log error with full traceback"""
        import traceback
        
        error_msg = f"❌ Error in {context}: {str(error)}"
        traceback_str = traceback.format_exc()
        
        self.error(error_msg)
        self.error(f"Traceback:\n{traceback_str}")
        
        event = {
            'timestamp': datetime.now().isoformat(),
            'event': 'error',
            'context': context,
            'error_message': str(error),
            'traceback': traceback_str
        }
        self.events_log.append(event)
    
    def save_logs_to_file(self):
        """Save metrics and events logs to JSON files"""
        # Save metrics log
        metrics_file = self.log_dir / f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics_log, f, indent=2, default=str)
        
        # Save events log
        events_file = self.log_dir / f"events_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(events_file, 'w') as f:
            json.dump(self.events_log, f, indent=2, default=str)
        
        self.info(f"📁 Logs saved to {metrics_file} and {events_file}")
    
    def _format_metrics(self, metrics: dict) -> str:
        """Format metrics for logging"""
        formatted = []
        for key, value in metrics.items():
            if isinstance(value, float):
                formatted.append(f"{key}={value:.4f}")
            else:
                formatted.append(f"{key}={value}")
        return ", ".join(formatted)
    
    def get_log_summary(self) -> dict:
        """Get summary of logged events"""
        summary = {
            'total_events': len(self.events_log),
            'total_metrics': len(self.metrics_log),
            'event_types': {},
            'last_activity': None
        }
        
        # Count event types
        for event in self.events_log:
            event_type = event.get('event', 'unknown')
            summary['event_types'][event_type] = summary['event_types'].get(event_type, 0) + 1
        
        # Last activity
        if self.events_log:
            summary['last_activity'] = self.events_log[-1]['timestamp']
        
        return summary

# Global logger instance
_global_logger = None

def get_logger(
    name: str = "deepfake_detector",
    log_dir: str = "logs",
    log_level: str = "INFO"
) -> DeepfakeLogger:
    """Get global logger instance"""
    global _global_logger
    
    if _global_logger is None:
        _global_logger = DeepfakeLogger(
            name=name,
            log_dir=log_dir,
            log_level=log_level
        )
    
    return _global_logger

def setup_experiment_logging(experiment_name: str, config: dict) -> DeepfakeLogger:
    """Setup logging for a specific experiment"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = f"logs/experiments/{experiment_name}_{timestamp}"
    
    logger = DeepfakeLogger(
        name=f"experiment_{experiment_name}",
        log_dir=log_dir,
        log_level="INFO"
    )
    
    logger.log_training_start(config)
    return logger