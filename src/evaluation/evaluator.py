import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import our custom modules
import sys
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.training.metrics import DeepfakeMetrics, MetricsTracker

class ModelEvaluator:
    """Comprehensive model evaluation for deepfake detection"""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        results_dir: str = "results"
    ):
        self.model = model
        self.device = device
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.results_dir / "metrics").mkdir(exist_ok=True)
        (self.results_dir / "visualizations").mkdir(exist_ok=True)
        (self.results_dir / "reports").mkdir(exist_ok=True)
        
    def evaluate_dataset(
        self,
        dataloader: DataLoader,
        dataset_name: str = "test"
    ) -> Dict[str, float]:
        """Evaluate model on a dataset"""
        print(f"Evaluating on {dataset_name} dataset...")
        
        self.model.eval()
        metrics = DeepfakeMetrics()
        
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(tqdm(dataloader, desc=f"Evaluating {dataset_name}")):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                
                # Store results
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                
                # Update metrics
                metrics.update(predictions, targets, probabilities)
        
        # Compute final metrics
        final_metrics = metrics.compute_all_metrics()
        
        # Save detailed results
        self._save_evaluation_results(
            dataset_name=dataset_name,
            metrics=final_metrics,
            predictions=all_predictions,
            targets=all_targets,
            probabilities=all_probabilities
        )
        
        # Generate visualizations
        self._generate_visualizations(metrics, dataset_name)
        
        # Print summary
        self._print_evaluation_summary(final_metrics, dataset_name)
        
        return final_metrics
    
    def evaluate_single_image(
        self,
        image_path: str,
        preprocessor: object,
        return_confidence: bool = True
    ) -> Dict[str, float]:
        """Evaluate a single image"""
        from PIL import Image
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        processed_image = preprocessor.preprocess_image(image, augment=False)
        
        # Add batch dimension
        batch_input = processed_image.unsqueeze(0).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(batch_input)
            probabilities = torch.softmax(outputs, dim=1)
            prediction = torch.argmax(outputs, dim=1)
        
        # Extract results
        fake_probability = probabilities[0, 1].item()
        real_probability = probabilities[0, 0].item()
        predicted_class = "fake" if prediction[0].item() == 1 else "real"
        
        results = {
            'predicted_class': predicted_class,
            'real_probability': real_probability,
            'fake_probability': fake_probability,
            'confidence': max(real_probability, fake_probability)
        }
        
        return results
    
    def batch_evaluate_images(
        self,
        image_paths: List[str],
        preprocessor: object,
        batch_size: int = 32
    ) -> List[Dict[str, float]]:
        """Evaluate multiple images in batches"""
        results = []
        
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Evaluating images"):
            batch_paths = image_paths[i:i + batch_size]
            
            for image_path in batch_paths:
                try:
                    result = self.evaluate_single_image(image_path, preprocessor)
                    result['image_path'] = image_path
                    results.append(result)
                except Exception as e:
                    print(f"Error evaluating {image_path}: {e}")
                    results.append({
                        'image_path': image_path,
                        'error': str(e)
                    })
        
        return results
    
    def cross_dataset_evaluation(
        self,
        dataloaders: Dict[str, DataLoader]
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate model across multiple datasets"""
        print("Starting cross-dataset evaluation...")
        
        all_results = {}
        
        for dataset_name, dataloader in dataloaders.items():
            print(f"\n{'='*50}")
            print(f"Evaluating on {dataset_name}")
            print(f"{'='*50}")
            
            results = self.evaluate_dataset(dataloader, dataset_name)
            all_results[dataset_name] = results
        
        # Generate comparison report
        self._generate_comparison_report(all_results)
        
        return all_results
    
    def analyze_failure_cases(
        self,
        dataloader: DataLoader,
        num_samples: int = 50
    ) -> Dict[str, List]:
        """Analyze model failure cases"""
        print("Analyzing failure cases...")
        
        self.model.eval()
        failure_cases = {
            'false_positives': [],  # Real images predicted as fake
            'false_negatives': []   # Fake images predicted as real
        }
        
        sample_count = 0
        
        with torch.no_grad():
            for inputs, targets in tqdm(dataloader, desc="Finding failure cases"):
                if sample_count >= num_samples:
                    break
                
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(inputs)
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                
                # Find misclassified samples
                misclassified = (predictions != targets)
                
                for i in range(len(inputs)):
                    if misclassified[i] and sample_count < num_samples:
                        sample_info = {
                            'predicted': predictions[i].item(),
                            'actual': targets[i].item(),
                            'confidence': torch.max(probabilities[i]).item(),
                            'probabilities': probabilities[i].cpu().numpy().tolist()
                        }
                        
                        if targets[i] == 0 and predictions[i] == 1:  # False positive
                            failure_cases['false_positives'].append(sample_info)
                        elif targets[i] == 1 and predictions[i] == 0:  # False negative
                            failure_cases['false_negatives'].append(sample_info)
                        
                        sample_count += 1
        
        # Save failure case analysis
        failure_report_path = self.results_dir / "reports" / "failure_analysis.json"
        with open(failure_report_path, 'w') as f:
            json.dump(failure_cases, f, indent=2)
        
        print(f"Failure analysis saved to {failure_report_path}")
        return failure_cases
    
    def _save_evaluation_results(
        self,
        dataset_name: str,
        metrics: Dict[str, float],
        predictions: List[int],
        targets: List[int],
        probabilities: List[List[float]]
    ):
        """Save detailed evaluation results"""
        results = {
            'dataset': dataset_name,
            'metrics': metrics,
            'predictions': predictions,
            'targets': targets,
            'probabilities': probabilities
        }
        
        # Save to JSON
        results_path = self.results_dir / "metrics" / f"{dataset_name}_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save metrics separately for easy access
        metrics_path = self.results_dir / "metrics" / f"{dataset_name}_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def _generate_visualizations(self, metrics: DeepfakeMetrics, dataset_name: str):
        """Generate evaluation visualizations"""
        viz_dir = self.results_dir / "visualizations"
        
        # Confusion matrix
        conf_matrix_path = viz_dir / f"{dataset_name}_confusion_matrix.png"
        metrics.plot_confusion_matrix(save_path=str(conf_matrix_path))
        
        # ROC curve
        roc_path = viz_dir / f"{dataset_name}_roc_curve.png"
        metrics.plot_roc_curve(save_path=str(roc_path))
        
        # Precision-Recall curve
        pr_path = viz_dir / f"{dataset_name}_pr_curve.png"
        metrics.plot_precision_recall_curve(save_path=str(pr_path))
        
        plt.close('all')  # Close all figures to free memory
    
    def _generate_comparison_report(self, all_results: Dict[str, Dict[str, float]]):
        """Generate comparison report across datasets"""
        report_path = self.results_dir / "reports" / "cross_dataset_comparison.md"
        
        with open(report_path, 'w') as f:
            f.write("# Cross-Dataset Evaluation Report\n\n")
            
            # Summary table
            f.write("## Performance Summary\n\n")
            f.write("| Dataset | Accuracy | Precision | Recall | F1-Score | AUC-ROC |\n")
            f.write("|---------|----------|-----------|--------|----------|----------|\n")
            
            for dataset_name, metrics in all_results.items():
                f.write(f"| {dataset_name} | ")
                f.write(f"{metrics.get('accuracy', 0):.4f} | ")
                f.write(f"{metrics.get('precision', 0):.4f} | ")
                f.write(f"{metrics.get('recall', 0):.4f} | ")
                f.write(f"{metrics.get('f1_score', 0):.4f} | ")
                f.write(f"{metrics.get('auc_roc', 0):.4f} |\n")
            
            # Detailed analysis
            f.write("\n## Detailed Analysis\n\n")
            for dataset_name, metrics in all_results.items():
                f.write(f"### {dataset_name}\n\n")
                for metric_name, value in metrics.items():
                    if isinstance(value, float):
                        f.write(f"- **{metric_name}**: {value:.4f}\n")
                f.write("\n")
    
    def _print_evaluation_summary(self, metrics: Dict[str, float], dataset_name: str):
        """Print evaluation summary"""
        print(f"\n{'='*50}")
        print(f"EVALUATION SUMMARY - {dataset_name.upper()}")
        print(f"{'='*50}")
        
        # Key metrics
        key_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
        
        for metric in key_metrics:
            if metric in metrics:
                print(f"{metric.upper():<12}: {metrics[metric]:.4f}")
        
        # Class-specific metrics
        print(f"\nCLASS-SPECIFIC METRICS:")
        print(f"Real Images  - Precision: {metrics.get('precision_real', 0):.4f}, Recall: {metrics.get('recall_real', 0):.4f}")
        print(f"Fake Images  - Precision: {metrics.get('precision_fake', 0):.4f}, Recall: {metrics.get('recall_fake', 0):.4f}")
        
        if 'optimal_threshold' in metrics:
            print(f"\nOPTIMAL THRESHOLD: {metrics['optimal_threshold']:.4f}")