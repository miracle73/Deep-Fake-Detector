import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import timm
from typing import Optional

class EfficientNetDeepfakeDetector(nn.Module):
    """
    EfficientNet-based deepfake detector for any type of AI-generated content
    Can detect: deepfake faces, AI art, generated scenes, synthetic objects, etc.
    """
    
    def __init__(
        self,
        model_name: str = "efficientnet-b4",
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.3,
        freeze_backbone: bool = False
    ):
        super(EfficientNetDeepfakeDetector, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.dropout = dropout
        
        # Load EfficientNet backbone
        if "efficientnet" in model_name.lower():
            self.backbone = timm.create_model(
                model_name, 
                pretrained=pretrained,
                num_classes=0,  # Remove classification head
                global_pool='avg'
            )
            # Get feature dimension
            self.feature_dim = self.backbone.num_features
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Custom classification head for deepfake detection
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout * 0.25),
            nn.Linear(128, num_classes)
        )
        
        # Initialize classifier weights
        self._initialize_classifier()
    
    def _initialize_classifier(self):
        """Initialize classifier weights"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass"""
        # Extract features using backbone
        features = self.backbone(x)
        
        # Apply classifier
        logits = self.classifier(features)
        
        return logits
    
    def predict_proba(self, x):
        """Get prediction probabilities"""
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
        return probabilities
    
    def predict(self, x, threshold: float = 0.5):
        """Get binary predictions"""
        probabilities = self.predict_proba(x)
        # Assuming class 1 is "fake"
        fake_prob = probabilities[:, 1]
        predictions = (fake_prob > threshold).long()
        return predictions, fake_prob
    
    def get_feature_vector(self, x):
        """Extract feature vector (useful for analysis)"""
        with torch.no_grad():
            features = self.backbone(x)
        return features

def create_efficientnet_detector(
    model_name: str = "efficientnet-b4",
    num_classes: int = 2,
    pretrained: bool = True,
    dropout: float = 0.3
) -> EfficientNetDeepfakeDetector:
    """Factory function to create EfficientNet detector"""
    return EfficientNetDeepfakeDetector(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout
    )