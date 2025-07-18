import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in deepfake detection
    Focuses training on hard examples
    """
    
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (N, C) predictions
            targets: (N,) ground truth labels
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing for better generalization
    Prevents overconfident predictions
    """
    
    def __init__(self, num_classes: int, smoothing: float = 0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (N, C) predictions
            target: (N,) ground truth labels
        """
        pred = pred.log_softmax(dim=-1)
        
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))

class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss for learning discriminative features
    Brings same-class samples closer, pushes different-class samples apart
    """
    
    def __init__(self, margin: float = 1.0, temperature: float = 0.1):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.temperature = temperature
    
    def forward(
        self, 
        features: torch.Tensor, 
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            features: (N, D) feature vectors
            labels: (N,) labels
        """
        # Normalize features
        features = F.normalize(features, p=2, dim=1)
        
        # Compute pairwise similarities
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Create label matrix
        labels = labels.unsqueeze(1)
        mask = torch.eq(labels, labels.T).float()
        
        # Remove diagonal (self-similarity)
        identity = torch.eye(mask.size(0), device=mask.device)
        mask = mask - identity
        
        # Compute contrastive loss
        exp_sim = torch.exp(similarity_matrix)
        sum_exp_sim = torch.sum(exp_sim * (1 - identity), dim=1, keepdim=True)
        
        positive_pairs = exp_sim * mask
        loss = -torch.log(positive_pairs / sum_exp_sim + 1e-8)
        
        # Average over positive pairs
        num_positive_pairs = torch.sum(mask, dim=1)
        loss = torch.sum(loss * mask, dim=1) / (num_positive_pairs + 1e-8)
        
        return torch.mean(loss)

class AdversarialLoss(nn.Module):
    """
    Adversarial training loss for robustness
    Makes model robust to small perturbations
    """
    
    def __init__(self, epsilon: float = 0.01, alpha: float = 0.5):
        super(AdversarialLoss, self).__init__()
        self.epsilon = epsilon
        self.alpha = alpha
        self.base_loss = nn.CrossEntropyLoss()
    
    def forward(
        self, 
        model: nn.Module, 
        inputs: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            model: The neural network model
            inputs: (N, C, H, W) input images
            targets: (N,) ground truth labels
        """
        # Standard loss
        outputs = model(inputs)
        standard_loss = self.base_loss(outputs, targets)
        
        # Generate adversarial examples
        inputs.requires_grad_(True)
        
        # Compute gradients
        grad_outputs = torch.ones_like(outputs)
        gradients = torch.autograd.grad(
            outputs=outputs,
            inputs=inputs,
            grad_outputs=grad_outputs,
            retain_graph=True,
            create_graph=True
        )[0]
        
        # Create adversarial perturbation
        perturbation = self.epsilon * torch.sign(gradients)
        adversarial_inputs = inputs + perturbation
        adversarial_inputs = torch.clamp(adversarial_inputs, 0, 1)
        
        # Adversarial loss
        adversarial_outputs = model(adversarial_inputs)
        adversarial_loss = self.base_loss(adversarial_outputs, targets)
        
        # Combined loss
        total_loss = (1 - self.alpha) * standard_loss + self.alpha * adversarial_loss
        
        return total_loss

class CombinedLoss(nn.Module):
    """
    Combined loss function for optimal deepfake detection
    """
    
    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.1,
        loss_weights: dict = None
    ):
        super(CombinedLoss, self).__init__()
        
        # Default loss weights
        if loss_weights is None:
            loss_weights = {
                'ce': 0.4,
                'focal': 0.3,
                'label_smooth': 0.3
            }
        
        self.loss_weights = loss_weights
        
        # Initialize individual losses
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.focal_loss = FocalLoss(alpha=class_weights, gamma=focal_gamma)
        self.label_smooth_loss = LabelSmoothingLoss(
            num_classes=2, 
            smoothing=label_smoothing
        )
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> dict:
        """
        Compute combined loss
        
        Returns:
            dict with individual losses and total loss
        """
        losses = {}
        
        # Individual losses
        losses['ce_loss'] = self.ce_loss(predictions, targets)
        losses['focal_loss'] = self.focal_loss(predictions, targets)
        losses['label_smooth_loss'] = self.label_smooth_loss(predictions, targets)
        
        # Combined loss
        total_loss = (
            self.loss_weights['ce'] * losses['ce_loss'] +
            self.loss_weights['focal'] * losses['focal_loss'] +
            self.loss_weights['label_smooth'] * losses['label_smooth_loss']
        )
        
        losses['total_loss'] = total_loss
        
        return losses