import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Dict, Any, Optional
import numpy as np

class VideoDeepfakeDetector(nn.Module):
    """Video deepfake detection model optimized for CPU training"""
    
    def __init__(
        self,
        frame_encoder: str = 'mobilenet_v2',
        temporal_model: str = 'lstm',
        num_frames: int = 30,
        num_classes: int = 2,
        dropout: float = 0.3,
        cpu_optimized: bool = True
    ):
        super().__init__()
        
        self.num_frames = num_frames
        self.num_classes = num_classes
        self.cpu_optimized = cpu_optimized
        
        # Frame encoder (lightweight for CPU)
        if frame_encoder == 'mobilenet_v2':
            backbone = models.mobilenet_v2(pretrained=True)
            self.frame_encoder = nn.Sequential(*list(backbone.features))
            feature_dim = 1280
        elif frame_encoder == 'resnet18':
            backbone = models.resnet18(pretrained=True)
            self.frame_encoder = nn.Sequential(*list(backbone.children())[:-2])
            feature_dim = 512
        else:
            raise ValueError(f"Unsupported encoder: {frame_encoder}")
        
        # Adaptive pooling for consistent output size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Temporal modeling
        if temporal_model == 'lstm':
            self.temporal_model = nn.LSTM(
                input_size=feature_dim,
                hidden_size=256,
                num_layers=2,
                batch_first=True,
                dropout=dropout,
                bidirectional=False  # Simpler for CPU
            )
            temporal_output_dim = 256
        elif temporal_model == 'gru':
            self.temporal_model = nn.GRU(
                input_size=feature_dim,
                hidden_size=256,
                num_layers=2,
                batch_first=True,
                dropout=dropout,
                bidirectional=False
            )
            temporal_output_dim = 256
        else:
            # Simple averaging fallback
            self.temporal_model = None
            temporal_output_dim = feature_dim
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(temporal_output_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize classifier weights"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass
        Args:
            x: Video tensor of shape (batch, channels, frames, height, width)
        Returns:
            Logits of shape (batch, num_classes)
        """
        batch_size, channels, num_frames, height, width = x.shape
        
        # Reshape for frame-wise processing: (batch*frames, channels, height, width)
        x = x.permute(0, 2, 1, 3, 4)  # (batch, frames, channels, height, width)
        x = x.reshape(batch_size * num_frames, channels, height, width)
        
        # Extract frame features
        frame_features = self.frame_encoder(x)  # (batch*frames, feature_dim, h', w')
        frame_features = self.adaptive_pool(frame_features)  # (batch*frames, feature_dim, 1, 1)
        frame_features = frame_features.flatten(1)  # (batch*frames, feature_dim)
        
        # Reshape back to sequence: (batch, frames, feature_dim)
        feature_dim = frame_features.shape[1]
        frame_features = frame_features.view(batch_size, num_frames, feature_dim)
        
        # Temporal modeling
        if self.temporal_model is not None:
            # LSTM/GRU
            temporal_output, _ = self.temporal_model(frame_features)
            # Use last output for classification
            video_features = temporal_output[:, -1, :]  # (batch, hidden_size)
        else:
            # Simple averaging
            video_features = torch.mean(frame_features, dim=1)  # (batch, feature_dim)
        
        # Classification
        logits = self.classifier(video_features)
        
        return logits
    
    def predict(self, frames: np.ndarray) -> Dict[str, Any]:
        """Predict on video frames"""
        self.eval()
        
        with torch.no_grad():
            # Preprocess frames
            if isinstance(frames, np.ndarray):
                # Convert to tensor and normalize
                frames_tensor = torch.from_numpy(frames).float() / 255.0
                frames_tensor = frames_tensor.permute(3, 0, 1, 2)  # (C, T, H, W)
                
                # Normalize
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
                frames_tensor = (frames_tensor - mean) / std
                
                # Add batch dimension
                frames_tensor = frames_tensor.unsqueeze(0)  # (1, C, T, H, W)
            
            # Forward pass
            logits = self.forward(frames_tensor)
            probabilities = F.softmax(logits, dim=1)
            
            # Extract results
            fake_prob = probabilities[0, 1].item()
            real_prob = probabilities[0, 0].item()
            is_fake = fake_prob > 0.5
            confidence = max(fake_prob, real_prob)
            
            return {
                'is_fake': is_fake,
                'fake_prob': fake_prob,
                'real_prob': real_prob,
                'confidence': confidence
            }
    
    def train_model(self, dataset, epochs: int = 10):
        """Simple training method (placeholder)"""
        # This would be implemented with proper training loop
        # For now, just a placeholder
        pass
    
    def save_model(self, path: str):
        """Save model state"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_config': {
                'num_frames': self.num_frames,
                'num_classes': self.num_classes
            }
        }, path)
    
    @classmethod
    def load_model(cls, path: str) -> 'VideoDeepfakeDetector':
        """Load model from saved state"""
        checkpoint = torch.load(path, map_location='cpu')
        config = checkpoint.get('model_config', {})
        
        model = cls(**config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model