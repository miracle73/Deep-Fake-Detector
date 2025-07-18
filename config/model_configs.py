import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class ModelConfig:
    """Model configuration class"""
    name: str
    num_classes: int
    pretrained: bool
    dropout: float

@dataclass
class TrainingConfig:
    """Training configuration class"""
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    patience: int

@dataclass
class DataConfig:
    """Data configuration class"""
    input_size: int
    train_split: float
    val_split: float
    test_split: float
    augmentation: bool

@dataclass
class PathConfig:
    """Path configuration class"""
    data_dir: str
    models_dir: str
    logs_dir: str
    results_dir: str

class ConfigManager:
    """Configuration manager for loading and managing configs"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = Path(config_path)
        self.config = self.load_config()
        
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            print(f"Config file not found: {self.config_path}")
            return self.get_default_config()
    
    def get_default_config(self) -> Dict[str, Any]:
        """Return default configuration if file not found"""
        return {
            'model': {
                'name': 'efficientnet-b4',
                'num_classes': 2,
                'pretrained': True,
                'dropout': 0.3
            },
            'training': {
                'batch_size': 32,
                'epochs': 50,
                'learning_rate': 0.001,
                'weight_decay': 0.0001,
                'patience': 10
            },
            'data': {
                'input_size': 224,
                'train_split': 0.7,
                'val_split': 0.2,
                'test_split': 0.1,
                'augmentation': True
            },
            'paths': {
                'data_dir': 'data/',
                'models_dir': 'checkpoints/',
                'logs_dir': 'logs/',
                'results_dir': 'results/'
            }
        }
    
    @property
    def model_config(self) -> ModelConfig:
        """Get model configuration"""
        model_cfg = self.config['model']
        return ModelConfig(**model_cfg)
    
    @property
    def training_config(self) -> TrainingConfig:
        """Get training configuration"""
        training_cfg = self.config['training']
        return TrainingConfig(**training_cfg)
    
    @property
    def data_config(self) -> DataConfig:
        """Get data configuration"""
        data_cfg = self.config['data']
        return DataConfig(**data_cfg)
    
    @property
    def path_config(self) -> PathConfig:
        """Get path configuration"""
        path_cfg = self.config['paths']
        return PathConfig(**path_cfg)