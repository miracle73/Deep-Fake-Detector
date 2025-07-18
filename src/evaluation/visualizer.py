import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from PIL import Image
import cv2
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class DeepfakeVisualizer:
    """Advanced visualization tools for deepfake detection analysis"""
    
    def __init__(self, save_dir: str = "results/visualizations"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Color schemes
        self.colors = {
            'real': '#2E8B57',      # Sea Green
            'fake': '#DC143C',      # Crimson
            'primary': '#1f77b4',   # Blue
            'secondary': '#ff7f0e', # Orange
            'success': '#2ca02c',   # Green
            'danger': '#d62728'     # Red
        }
    
    def plot_training_history(
        self,
        history: Dict[str, List[float]],
        save_name: str = "training_history.png"
    ) -> plt.Figure:
        """Plot comprehensive training history"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Training History Overview', fontsize=16, fontweight='bold')
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Loss plot
        axes[0, 0].plot(epochs, history['train_loss'], label='Training Loss', color=self.colors['primary'])
        axes[0, 0].plot(epochs, history['val_loss'], label='Validation Loss', color=self.colors['secondary'])
        axes[0, 0].set_title('Training & Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy plot
        axes[0, 1].plot(epochs, history['train_acc'], label='Training Accuracy', color=self.colors['primary'])
        axes[0, 1].plot(epochs, history['val_acc'], label='Validation Accuracy', color=self.colors['secondary'])
        axes[0, 1].set_title('Training & Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rate plot
        if 'learning_rates' in history:
            axes[0, 2].plot(epochs, history['learning_rates'], color=self.colors['danger'])
            axes[0, 2].set_title('Learning Rate Schedule')
            axes[0, 2].set_xlabel('Epoch')
            axes[0, 2].set_ylabel('Learning Rate')
            axes[0, 2].set_yscale('log')
            axes[0, 2].grid(True, alpha=0.3)
        
        # Loss difference (overfitting indicator)
        loss_diff = np.array(history['val_loss']) - np.array(history['train_loss'])
        axes[1, 0].plot(epochs, loss_diff, color=self.colors['danger'], alpha=0.7)
        axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 0].set_title('Validation - Training Loss (Overfitting Indicator)')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss Difference')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Moving average of validation accuracy
        if len(history['val_acc']) > 5:
            window = min(5, len(history['val_acc']) // 4)
            val_acc_smooth = pd.Series(history['val_acc']).rolling(window=window).mean()
            axes[1, 1].plot(epochs, history['val_acc'], alpha=0.3, color=self.colors['secondary'])
            axes[1, 1].plot(epochs, val_acc_smooth, color=self.colors['secondary'], linewidth=2)
            axes[1, 1].set_title(f'Validation Accuracy (Smoothed, window={window})')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Accuracy')
            axes[1, 1].grid(True, alpha=0.3)
        
        # Best metrics summary
        axes[1, 2].axis('off')
        best_val_acc = max(history['val_acc'])
        best_val_acc_epoch = history['val_acc'].index(best_val_acc) + 1
        min_val_loss = min(history['val_loss'])
        min_val_loss_epoch = history['val_loss'].index(min_val_loss) + 1
        
        summary_text = f"""
        Best Metrics:
        
        Best Validation Accuracy: {best_val_acc:.4f}
        (Epoch {best_val_acc_epoch})
        
        Lowest Validation Loss: {min_val_loss:.4f}
        (Epoch {min_val_loss_epoch})
        
        Final Training Accuracy: {history['train_acc'][-1]:.4f}
        Final Validation Accuracy: {history['val_acc'][-1]:.4f}
        
        Total Epochs: {len(epochs)}
        """
        
        axes[1, 2].text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        
        # Save figure
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_model_comparison(
        self,
        models_metrics: Dict[str, Dict[str, float]],
        save_name: str = "model_comparison.png"
    ) -> plt.Figure:
        """Compare multiple models performance"""
        
        # Prepare data
        models = list(models_metrics.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
        
        # Create DataFrame
        data = []
        for model in models:
            for metric in metrics:
                if metric in models_metrics[model]:
                    data.append({
                        'Model': model,
                        'Metric': metric.replace('_', ' ').title(),
                        'Value': models_metrics[model][metric]
                    })
        
        df = pd.DataFrame(data)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # Bar plot comparison
        pivot_df = df.pivot(index='Model', columns='Metric', values='Value')
        pivot_df.plot(kind='bar', ax=axes[0, 0], rot=45)
        axes[0, 0].set_title('All Metrics Comparison')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Radar chart
        if len(models) <= 4:  # Only for reasonable number of models
            angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
            angles += angles[:1]  # Complete the circle
            
            axes[0, 1].set_theta_offset(np.pi / 2)
            axes[0, 1].set_theta_direction(-1)
            axes[0, 1].set_thetagrids(np.degrees(angles[:-1]), metrics)
            
            for i, model in enumerate(models):
                values = [models_metrics[model].get(metric, 0) for metric in metrics]
                values += values[:1]  # Complete the circle
                
                color = plt.cm.Set1(i / len(models))
                axes[0, 1].plot(angles, values, 'o-', linewidth=2, label=model, color=color)
                axes[0, 1].fill(angles, values, alpha=0.25, color=color)
            
            axes[0, 1].set_ylim(0, 1)
            axes[0, 1].set_title('Performance Radar Chart')
            axes[0, 1].legend(loc='upper right', bbox_to_anchor=(1.2, 1))
        else:
            axes[0, 1].text(0.5, 0.5, 'Too many models\nfor radar chart', 
                           ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Radar Chart (Unavailable)')
        
        # Accuracy ranking
        accuracy_data = [(model, metrics_dict.get('accuracy', 0)) 
                        for model, metrics_dict in models_metrics.items()]
        accuracy_data.sort(key=lambda x: x[1], reverse=True)
        
        models_sorted, accuracies = zip(*accuracy_data)
        bars = axes[1, 0].bar(models_sorted, accuracies, color=self.colors['primary'])
        axes[1, 0].set_title('Accuracy Ranking')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{acc:.3f}', ha='center', va='bottom')
        
        # Performance summary table
        axes[1, 1].axis('off')
        
        # Create table data
        table_data = []
        for model in models:
            row = [model]
            for metric in metrics:
                value = models_metrics[model].get(metric, 0)
                row.append(f'{value:.3f}')
            table_data.append(row)
        
        table = axes[1, 1].table(cellText=table_data,
                                colLabels=['Model'] + [m.replace('_', ' ').title() for m in metrics],
                                cellLoc='center',
                                loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        axes[1, 1].set_title('Performance Summary Table')
        
        plt.tight_layout()
        
        # Save figure
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_prediction_examples(
        self,
        images: List[np.ndarray],
        predictions: List[str],
        confidences: List[float],
        true_labels: Optional[List[str]] = None,
        probabilities: Optional[List[np.ndarray]] = None,
        save_name: str = "prediction_examples.png",
        max_images: int = 16
    ) -> plt.Figure:
        """Plot prediction examples with detailed information"""
        
        n_images = min(len(images), max_images)
        cols = 4
        rows = (n_images + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle('Prediction Examples', fontsize=16, fontweight='bold')
        
        for i in range(n_images):
            row = i // cols
            col = i % cols
            ax = axes[row, col]
            
            # Display image
            ax.imshow(images[i])
            ax.axis('off')
            
            # Determine colors based on correctness
            pred_color = self.colors['fake'] if predictions[i] == 'fake' else self.colors['real']
            
            if true_labels:
                is_correct = predictions[i] == true_labels[i]
                border_color = self.colors['success'] if is_correct else self.colors['danger']
                
                # Add colored border
                for spine in ax.spines.values():
                    spine.set_edgecolor(border_color)
                    spine.set_linewidth(3)
                    spine.set_visible(True)
            
            # Create detailed title
            title_parts = []
            title_parts.append(f"Pred: {predictions[i].upper()}")
            title_parts.append(f"Conf: {confidences[i]:.3f}")
            
            if true_labels:
                title_parts.append(f"True: {true_labels[i].upper()}")
                status = "✓" if predictions[i] == true_labels[i] else "✗"
                title_parts.append(status)
            
            if probabilities:
                real_prob = probabilities[i][0] if len(probabilities[i]) > 1 else 1 - probabilities[i][0]
                fake_prob = probabilities[i][1] if len(probabilities[i]) > 1 else probabilities[i][0]
                title_parts.append(f"R:{real_prob:.2f} F:{fake_prob:.2f}")
            
            title = '\n'.join([' | '.join(title_parts[:2]), ' | '.join(title_parts[2:])])
            ax.set_title(title, fontsize=10, color=pred_color, fontweight='bold')
        
        # Hide empty subplots
        for i in range(n_images, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        
        # Save figure
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_confidence_distribution(
        self,
        real_confidences: List[float],
        fake_confidences: List[float],
        save_name: str = "confidence_distribution.png"
    ) -> plt.Figure:
        """Plot confidence distribution for real vs fake predictions"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Confidence Distribution Analysis', fontsize=16, fontweight='bold')
        
        # Histogram overlay
        axes[0, 0].hist(real_confidences, bins=30, alpha=0.7, label='Real Images', 
                       color=self.colors['real'], density=True)
        axes[0, 0].hist(fake_confidences, bins=30, alpha=0.7, label='Fake Images', 
                       color=self.colors['fake'], density=True)
        axes[0, 0].set_title('Confidence Distribution Overlay')
        axes[0, 0].set_xlabel('Confidence Score')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Box plot comparison
        data_for_box = [real_confidences, fake_confidences]
        box_plot = axes[0, 1].boxplot(data_for_box, labels=['Real', 'Fake'], patch_artist=True)
        box_plot['boxes'][0].set_facecolor(self.colors['real'])
        box_plot['boxes'][1].set_facecolor(self.colors['fake'])
        axes[0, 1].set_title('Confidence Distribution Comparison')
        axes[0, 1].set_ylabel('Confidence Score')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Cumulative distribution
        real_sorted = np.sort(real_confidences)
        fake_sorted = np.sort(fake_confidences)
        real_cum = np.arange(1, len(real_sorted) + 1) / len(real_sorted)
        fake_cum = np.arange(1, len(fake_sorted) + 1) / len(fake_sorted)
        
        axes[1, 0].plot(real_sorted, real_cum, label='Real Images', 
                       color=self.colors['real'], linewidth=2)
        axes[1, 0].plot(fake_sorted, fake_cum, label='Fake Images', 
                       color=self.colors['fake'], linewidth=2)
        axes[1, 0].set_title('Cumulative Distribution Function')
        axes[1, 0].set_xlabel('Confidence Score')
        axes[1, 0].set_ylabel('Cumulative Probability')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Statistics summary
        axes[1, 1].axis('off')
        
        real_stats = {
            'mean': np.mean(real_confidences),
            'std': np.std(real_confidences),
            'median': np.median(real_confidences),
            'min': np.min(real_confidences),
            'max': np.max(real_confidences)
        }
        
        fake_stats = {
            'mean': np.mean(fake_confidences),
            'std': np.std(fake_confidences),
            'median': np.median(fake_confidences),
            'min': np.min(fake_confidences),
            'max': np.max(fake_confidences)
        }
        
        stats_text = f"""
        Confidence Statistics:
        
        Real Images:
        Mean: {real_stats['mean']:.3f}
        Std: {real_stats['std']:.3f}
        Median: {real_stats['median']:.3f}
        Range: [{real_stats['min']:.3f}, {real_stats['max']:.3f}]
        
        Fake Images:
        Mean: {fake_stats['mean']:.3f}
        Std: {fake_stats['std']:.3f}
        Median: {fake_stats['median']:.3f}
        Range: [{fake_stats['min']:.3f}, {fake_stats['max']:.3f}]
        """
        
        axes[1, 1].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        
        # Save figure
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_interactive_dashboard(
        self,
        results_data: Dict,
        save_name: str = "interactive_dashboard.html"
    ) -> str:
        """Create interactive dashboard using Plotly"""
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Training Loss', 'Training Accuracy', 
                          'ROC Curve', 'Precision-Recall Curve',
                          'Confusion Matrix', 'Model Comparison'),
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "heatmap"}, {"type": "bar"}]]
        )
        
        # Training history plots (if available)
        if 'training_history' in results_data:
            history = results_data['training_history']
            epochs = list(range(1, len(history['train_loss']) + 1))
            
            # Training loss
            fig.add_trace(
                go.Scatter(x=epochs, y=history['train_loss'], name='Train Loss', 
                          line=dict(color=self.colors['primary'])),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=epochs, y=history['val_loss'], name='Val Loss', 
                          line=dict(color=self.colors['secondary'])),
                row=1, col=1
            )
            
            # Training accuracy
            fig.add_trace(
                go.Scatter(x=epochs, y=history['train_acc'], name='Train Acc', 
                          line=dict(color=self.colors['primary'])),
                row=1, col=2
            )
            fig.add_trace(
                go.Scatter(x=epochs, y=history['val_acc'], name='Val Acc', 
                          line=dict(color=self.colors['secondary'])),
                row=1, col=2
            )
        
        # ROC curve (if available)
        if 'roc_data' in results_data:
            roc_data = results_data['roc_data']
            fig.add_trace(
                go.Scatter(x=roc_data['fpr'], y=roc_data['tpr'], 
                          name=f'ROC (AUC={roc_data["auc"]:.3f})',
                          line=dict(color=self.colors['primary'])),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(x=[0, 1], y=[0, 1], name='Random', 
                          line=dict(dash='dash', color='gray')),
                row=2, col=1
            )
        
        # Precision-Recall curve (if available)
        if 'pr_data' in results_data:
            pr_data = results_data['pr_data']
            fig.add_trace(
                go.Scatter(x=pr_data['recall'], y=pr_data['precision'], 
                          name=f'PR (AP={pr_data["ap"]:.3f})',
                          line=dict(color=self.colors['primary'])),
                row=2, col=2
            )
        
        # Confusion matrix (if available)
        if 'confusion_matrix' in results_data:
            cm = results_data['confusion_matrix']
            fig.add_trace(
                go.Heatmap(z=cm, x=['Real', 'Fake'], y=['Real', 'Fake'],
                          colorscale='Blues', showscale=False,
                          text=cm, texttemplate="%{text}", textfont={"size": 16}),
                row=3, col=1
            )
        
        # Model comparison (if available)
        if 'model_comparison' in results_data:
            comp_data = results_data['model_comparison']
            models = list(comp_data.keys())
            accuracies = [comp_data[model].get('accuracy', 0) for model in models]
            
            fig.add_trace(
                go.Bar(x=models, y=accuracies, name='Accuracy',
                      marker_color=self.colors['primary']),
                row=3, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=1200,
            title_text="Deepfake Detection Dashboard",
            title_x=0.5,
            showlegend=True
        )
        
        # Save interactive plot
        save_path = self.save_dir / save_name
        fig.write_html(str(save_path))
        
        return str(save_path)
