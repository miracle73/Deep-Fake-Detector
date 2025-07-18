import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    average_precision_score, roc_curve, precision_recall_curve
)
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns

class DeepfakeMetrics:
    """Comprehensive metrics for deepfake detection evaluation"""
    
    def __init__(self, num_classes: int = 2):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.predictions = []
        self.targets = []
        self.probabilities = []
        
    def update(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor, 
        probabilities: Optional[torch.Tensor] = None
    ):
        """Update metrics with new batch"""
        # Convert to numpy
        pred_np = predictions.detach().cpu().numpy()
        target_np = targets.detach().cpu().numpy()
        
        self.predictions.extend(pred_np)
        self.targets.extend(target_np)
        
        if probabilities is not None:
            prob_np = probabilities.detach().cpu().numpy()
            if len(prob_np.shape) > 1:
                # Take probability of positive class (fake)
                prob_np = prob_np[:, 1] if prob_np.shape[1] > 1 else prob_np[:, 0]
            self.probabilities.extend(prob_np)
    
    def compute_basic_metrics(self) -> Dict[str, float]:
        """Compute basic classification metrics"""
        if not self.predictions:
            return {}
        
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        
        metrics = {
            'accuracy': accuracy_score(targets, predictions),
            'precision': precision_score(targets, predictions, average='binary'),
            'recall': recall_score(targets, predictions, average='binary'),
            'f1_score': f1_score(targets, predictions, average='binary'),
            'specificity': self._compute_specificity(targets, predictions),
        }
        
        # Add per-class metrics
        precision_per_class = precision_score(targets, predictions, average=None)
        recall_per_class = recall_score(targets, predictions, average=None)
        f1_per_class = f1_score(targets, predictions, average=None)
        
        metrics.update({
            'precision_real': precision_per_class[0],
            'precision_fake': precision_per_class[1],
            'recall_real': recall_per_class[0],
            'recall_fake': recall_per_class[1],
            'f1_real': f1_per_class[0],
            'f1_fake': f1_per_class[1],
        })
        
        return metrics
    
    def compute_roc_metrics(self) -> Dict[str, float]:
        """Compute ROC-based metrics"""
        if not self.probabilities:
            return {}
        
        targets = np.array(self.targets)
        probabilities = np.array(self.probabilities)
        
        try:
            auc_score = roc_auc_score(targets, probabilities)
            ap_score = average_precision_score(targets, probabilities)
            
            # Find optimal threshold
            fpr, tpr, thresholds = roc_curve(targets, probabilities)
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]
            
            return {
                'auc_roc': auc_score,
                'average_precision': ap_score,
                'optimal_threshold': optimal_threshold,
                'optimal_tpr': tpr[optimal_idx],
                'optimal_fpr': fpr[optimal_idx]
            }
        except Exception as e:
            print(f"Error computing ROC metrics: {e}")
            return {}
    
    def compute_confusion_matrix(self) -> np.ndarray:
        """Compute confusion matrix"""
        if not self.predictions:
            return np.array([])
        
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        
        return confusion_matrix(targets, predictions)
    
    def compute_all_metrics(self) -> Dict[str, float]:
        """Compute all available metrics"""
        metrics = {}
        metrics.update(self.compute_basic_metrics())
        metrics.update(self.compute_roc_metrics())
        
        return metrics
    
    def _compute_specificity(self, targets: np.ndarray, predictions: np.ndarray) -> float:
        """Compute specificity (true negative rate)"""
        cm = confusion_matrix(targets, predictions)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            return tn / (tn + fp) if (tn + fp) > 0 else 0.0
        return 0.0
    
    def get_classification_report(self) -> str:
        """Get detailed classification report"""
        if not self.predictions:
            return "No predictions available"
        
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        
        class_names = ['Real', 'Fake']
        return classification_report(
            targets, predictions, 
            target_names=class_names,
            digits=4
        )
    
    def plot_confusion_matrix(self, save_path: Optional[str] = None) -> plt.Figure:
        """Plot confusion matrix"""
        cm = self.compute_confusion_matrix()
        
        if cm.size == 0:
            return None
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create heatmap
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Real', 'Fake'],
            yticklabels=['Real', 'Fake'],
            ax=ax
        )
        
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_roc_curve(self, save_path: Optional[str] = None) -> plt.Figure:
        """Plot ROC curve"""
        if not self.probabilities:
            return None
        
        targets = np.array(self.targets)
        probabilities = np.array(self.probabilities)
        
        fpr, tpr, _ = roc_curve(targets, probabilities)
        auc_score = roc_auc_score(targets, probabilities)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc_score:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
        
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_precision_recall_curve(self, save_path: Optional[str] = None) -> plt.Figure:
        """Plot Precision-Recall curve"""
        if not self.probabilities:
            return None
        
        targets = np.array(self.targets)
        probabilities = np.array(self.probabilities)
        
        precision, recall, _ = precision_recall_curve(targets, probabilities)
        ap_score = average_precision_score(targets, probabilities)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(recall, precision, linewidth=2, 
                label=f'PR Curve (AP = {ap_score:.3f})')
        
        # Add baseline
        baseline = np.sum(targets) / len(targets)
        ax.axhline(y=baseline, color='k', linestyle='--', alpha=0.5,
                   label=f'Baseline (AP = {baseline:.3f})')
        
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig

class MetricsTracker:
    """Track metrics during training"""
    
    def __init__(self):
        self.train_metrics = []
        self.val_metrics = []
        
    def add_train_metrics(self, metrics: Dict[str, float]):
        """Add training metrics for current epoch"""
        self.train_metrics.append(metrics)
    
    def add_val_metrics(self, metrics: Dict[str, float]):
        """Add validation metrics for current epoch"""
        self.val_metrics.append(metrics)
    
    def plot_training_curves(self, save_path: Optional[str] = None) -> plt.Figure:
        """Plot training curves"""
        if not self.train_metrics or not self.val_metrics:
            return None
        
        epochs = range(1, len(self.train_metrics) + 1)
        
        # Extract metrics
        train_loss = [m.get('loss', 0) for m in self.train_metrics]
        val_loss = [m.get('loss', 0) for m in self.val_metrics]
        train_acc = [m.get('accuracy', 0) for m in self.train_metrics]
        val_acc = [m.get('accuracy', 0) for m in self.val_metrics]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(epochs, train_loss, 'b-', label='Training Loss')
        ax1.plot(epochs, val_loss, 'r-', label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2.plot(epochs, train_acc, 'b-', label='Training Accuracy')
        ax2.plot(epochs, val_acc, 'r-', label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
