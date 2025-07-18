#!/usr/bin/env python3
"""
Model evaluation script for deepfake detection
"""

import sys
import os
from pathlib import Path
import torch
import argparse
import json

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config.model_configs import ConfigManager
from models.efficientnet_detector import create_efficientnet_detector
from src.data.preprocessing import DeepfakePreprocessor
from src.data.dataset import create_data_loaders
from src.evaluation.evaluator import ModelEvaluator

def load_trained_model(checkpoint_path: str, config: dict, device: torch.device):
    """Load trained model from checkpoint"""
    print(f"Loading model from {checkpoint_path}")
    
    # Create model
    model = create_efficientnet_detector(
        model_name=config['model']['name'],
        num_classes=config['model']['num_classes'],
        pretrained=config['model']['pretrained'],
        dropout=config['model']['dropout']
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"✅ Model loaded successfully")
    return model

def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Evaluate Deepfake Detection Model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Path to data directory')
    parser.add_argument('--results-dir', type=str, default='results',
                        help='Path to save results')
    parser.add_argument('--dataset', type=str, choices=['train', 'val', 'test', 'all'],
                        default='test', help='Dataset to evaluate on')
    parser.add_argument('--single-image', type=str, default=None,
                        help='Path to single image for evaluation')
    parser.add_argument('--failure-analysis', action='store_true',
                        help='Perform failure case analysis')
    
    args = parser.parse_args()
    
    print("🔍 DEEPFAKE DETECTION MODEL EVALUATION")
    print("=" * 50)
    
    # Check if checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        return
    
    # Load configuration
    print("📝 Loading configuration...")
    config_manager = ConfigManager(args.config)
    config = config_manager.config
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}")
    
    # Load model
    model = load_trained_model(args.checkpoint, config, device)
    
    # Create evaluator
    evaluator = ModelEvaluator(model, device, args.results_dir)
    
    # Single image evaluation
    if args.single_image:
        print(f"\n🖼️ Evaluating single image: {args.single_image}")
        
        if not Path(args.single_image).exists():
            print(f"❌ Image not found: {args.single_image}")
            return
        
        # Create preprocessor
        preprocessor = DeepfakePreprocessor(
            input_size=config['data']['input_size'],
            normalize=True,
            augment=False
        )
        
        # Evaluate
        result = evaluator.evaluate_single_image(args.single_image, preprocessor)
        
        print(f"\n📊 Results:")
        print(f"   Predicted Class: {result['predicted_class'].upper()}")
        print(f"   Confidence: {result['confidence']:.4f}")
        print(f"   Real Probability: {result['real_probability']:.4f}")
        print(f"   Fake Probability: {result['fake_probability']:.4f}")
        
        return
    
    # Dataset evaluation
    print(f"\n📊 Loading dataset from {args.data_dir}...")
    
    # Create preprocessor
    preprocessor = DeepfakePreprocessor(
        input_size=config['data']['input_size'],
        normalize=True,
        augment=False
    )
    
    try:
        # Load data
        train_loader, val_loader, test_loader = create_data_loaders(
            data_dir=args.data_dir,
            preprocessor=preprocessor,
            batch_size=config['training']['batch_size'],
            num_workers=4,
            balance_classes=False  # Don't balance for evaluation
        )
        
        # Select datasets to evaluate
        datasets_to_eval = {}
        
        if args.dataset == 'all':
            datasets_to_eval = {
                'train': train_loader,
                'val': val_loader,
                'test': test_loader
            }
        elif args.dataset == 'train':
            datasets_to_eval = {'train': train_loader}
        elif args.dataset == 'val':
            datasets_to_eval = {'val': val_loader}
        elif args.dataset == 'test':
            datasets_to_eval = {'test': test_loader}
        
        # Perform evaluation
        if len(datasets_to_eval) > 1:
            # Cross-dataset evaluation
            all_results = evaluator.cross_dataset_evaluation(datasets_to_eval)
            
            # Print summary
            print(f"\n📈 CROSS-DATASET SUMMARY")
            print("=" * 50)
            for dataset_name, metrics in all_results.items():
                print(f"{dataset_name:<8}: Acc={metrics.get('accuracy', 0):.4f}, "
                      f"F1={metrics.get('f1_score', 0):.4f}, "
                      f"AUC={metrics.get('auc_roc', 0):.4f}")
        
        else:
            # Single dataset evaluation
            dataset_name, dataloader = list(datasets_to_eval.items())[0]
            results = evaluator.evaluate_dataset(dataloader, dataset_name)
        
        # Failure analysis
        if args.failure_analysis:
            print(f"\n🔍 Performing failure case analysis...")
            failure_cases = evaluator.analyze_failure_cases(test_loader)
            
            print(f"Found {len(failure_cases['false_positives'])} false positives")
            print(f"Found {len(failure_cases['false_negatives'])} false negatives")
        
        print(f"\n✅ Evaluation completed!")
        print(f"📁 Results saved to: {args.results_dir}")
        print(f"   📊 Metrics: {args.results_dir}/metrics/")
        print(f"   📈 Visualizations: {args.results_dir}/visualizations/")
        print(f"   📄 Reports: {args.results_dir}/reports/")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        raise

if __name__ == "__main__":
    main()