import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional

class ResNetDeepfakeDetector(nn.Module):
    """
    ResNet-based deepfake detector
    Alternative to EfficientNet for comparison and ensemble use
    """
    
    def __init__(
        self,
        model_name: str = "resnet50",
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.3,
        freeze_backbone: bool = False
    ):
        super(ResNetDeepfakeDetector, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.dropout = dropout
        
        # Load ResNet backbone
        if model_name == "resnet18":
            self.backbone = models.resnet18(pretrained=pretrained)
            self.feature_dim = 512
        elif model_name == "resnet34":
            self.backbone = models.resnet34(pretrained=pretrained)
            self.feature_dim = 512
        elif model_name == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
            self.feature_dim = 2048
        elif model_name == "resnet101":
            self.backbone = models.resnet101(pretrained=pretrained)
            self.feature_dim = 2048
        elif model_name == "resnet152":
            self.backbone = models.resnet152(pretrained=pretrained)
            self.feature_dim = 2048
        else:
            raise ValueError(f"Unsupported ResNet model: {model_name}")
        
        # Remove the original classifier
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Custom classification head for deepfake detection
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Dropout(dropout * 0.5),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout * 0.25),
            nn.Linear(256, num_classes)
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
        fake_prob = probabilities[:, 1]
        predictions = (fake_prob > threshold).long()
        return predictions, fake_prob
    
    def get_feature_vector(self, x):
        """Extract feature vector (useful for analysis)"""
        with torch.no_grad():
            features = self.backbone(x)
            # Global average pooling
            features = F.adaptive_avg_pool2d(features, (1, 1))
            features = features.flatten(1)
        return features

def create_resnet_detector(
    model_name: str = "resnet50",
    num_classes: int = 2,
    pretrained: bool = True,
    dropout: float = 0.3
) -> ResNetDeepfakeDetector:
    """Factory function to create ResNet detector"""
    return ResNetDeepfakeDetector(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout
    )