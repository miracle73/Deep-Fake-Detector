import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Union
import numpy as np

from .efficientnet_detector import EfficientNetDeepfakeDetector
from .resnet_detector import ResNetDeepfakeDetector

class EnsembleDeepfakeDetector(nn.Module):
    """
    Ensemble deepfake detector combining multiple models
    Combines predictions from different architectures for better performance
    """
    
    def __init__(
        self,
        models: List[nn.Module],
        model_weights: Optional[List[float]] = None,
        ensemble_method: str = "voting",  # "voting", "averaging", "learned"
        num_classes: int = 2
    ):
        super(EnsembleDeepfakeDetector, self).__init__()
        
        self.models = nn.ModuleList(models)
        self.num_models = len(models)
        self.num_classes = num_classes
        self.ensemble_method = ensemble_method
        
        # Set model weights
        if model_weights is None:
            self.model_weights = [1.0 / self.num_models] * self.num_models
        else:
            assert len(model_weights) == self.num_models, "Number of weights must match number of models"
            # Normalize weights
            total_weight = sum(model_weights)
            self.model_weights = [w / total_weight for w in model_weights]
        
        # For learned ensemble, add a meta-classifier
        if ensemble_method == "learned":
            # Feature dimension = num_models * num_classes (concatenated predictions)
            feature_dim = self.num_models * num_classes
            self.meta_classifier = nn.Sequential(
                nn.Linear(feature_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, num_classes)
            )
    
    def forward(self, x):
        """Forward pass through ensemble"""
        if self.ensemble_method == "voting":
            return self._voting_forward(x)
        elif self.ensemble_method == "averaging":
            return self._averaging_forward(x)
        elif self.ensemble_method == "learned":
            return self._learned_forward(x)
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
    
    def _voting_forward(self, x):
        """Hard voting ensemble"""
        predictions = []
        
        for model in self.models:
            with torch.no_grad():
                logits = model(x)
                pred = torch.argmax(F.softmax(logits, dim=1), dim=1)
                predictions.append(pred)
        
        # Stack predictions and find majority vote
        predictions = torch.stack(predictions, dim=1)  # (batch_size, num_models)
        
        # Count votes for each class
        batch_size = x.size(0)
        final_predictions = torch.zeros(batch_size, self.num_classes, device=x.device)
        
        for i in range(self.num_classes):
            votes = (predictions == i).sum(dim=1).float()
            final_predictions[:, i] = votes
        
        return final_predictions
    
    def _averaging_forward(self, x):
        """Weighted averaging ensemble"""
        weighted_logits = None
        
        for i, model in enumerate(self.models):
            logits = model(x)
            weight = self.model_weights[i]
            
            if weighted_logits is None:
                weighted_logits = weight * logits
            else:
                weighted_logits += weight * logits
        
        return weighted_logits
    
    def _learned_forward(self, x):
        """Learned ensemble with meta-classifier"""
        # Get predictions from all models
        model_predictions = []
        
        for model in self.models:
            logits = model(x)
            probabilities = F.softmax(logits, dim=1)
            model_predictions.append(probabilities)
        
        # Concatenate all predictions
        concatenated_features = torch.cat(model_predictions, dim=1)
        
        # Pass through meta-classifier
        final_logits = self.meta_classifier(concatenated_features)
        
        return final_logits
    
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
    
    def get_individual_predictions(self, x):
        """Get predictions from individual models"""
        individual_preds = {}
        
        for i, model in enumerate(self.models):
            with torch.no_grad():
                logits = model(x)
                probabilities = F.softmax(logits, dim=1)
                individual_preds[f'model_{i}'] = probabilities
        
        return individual_preds
    
    def get_prediction_confidence_analysis(self, x):
        """Analyze prediction confidence across models"""
        individual_preds = self.get_individual_predictions(x)
        ensemble_pred = self.predict_proba(x)
        
        # Calculate agreement metrics
        fake_probs = [pred[:, 1] for pred in individual_preds.values()]
        fake_probs_tensor = torch.stack(fake_probs, dim=1)
        
        # Standard deviation across models (lower = more agreement)
        agreement = torch.std(fake_probs_tensor, dim=1)
        
        # Average confidence across models
        avg_confidence = torch.mean(torch.max(torch.stack(list(individual_preds.values()), dim=1), dim=2)[0], dim=1)
        
        analysis = {
            'ensemble_prediction': ensemble_pred,
            'individual_predictions': individual_preds,
            'prediction_agreement': agreement,  # Lower values = better agreement
            'average_confidence': avg_confidence,
            'ensemble_confidence': torch.max(ensemble_pred, dim=1)[0]
        }
        
        return analysis

def create_efficientnet_resnet_ensemble(
    efficientnet_checkpoint: Optional[str] = None,
    resnet_checkpoint: Optional[str] = None,
    ensemble_method: str = "averaging",
    model_weights: Optional[List[float]] = None
) -> EnsembleDeepfakeDetector:
    """Create an ensemble of EfficientNet and ResNet models"""
    
    # Create models
    efficientnet = EfficientNetDeepfakeDetector(model_name="efficientnet-b4")
    resnet = ResNetDeepfakeDetector(model_name="resnet50")
    
    # Load checkpoints if provided
    if efficientnet_checkpoint:
        checkpoint = torch.load(efficientnet_checkpoint, map_location='cpu')
        efficientnet.load_state_dict(checkpoint['model_state_dict'])
    
    if resnet_checkpoint:
        checkpoint = torch.load(resnet_checkpoint, map_location='cpu')
        resnet.load_state_dict(checkpoint['model_state_dict'])
    
    # Create ensemble
    ensemble = EnsembleDeepfakeDetector(
        models=[efficientnet, resnet],
        model_weights=model_weights,
        ensemble_method=ensemble_method
    )
    
    return ensemble