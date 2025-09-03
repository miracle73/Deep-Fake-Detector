#!/usr/bin/env python3
"""
Evaluate Audio Deepfake Detection Model
"""

import sys
from pathlib import Path
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent / "src"))

from audio_model import AudioDeepfakeDetector
from audio_dataset import create_data_loaders

def evaluate_model(model_path: str, data_dir: str):
    """Evaluate trained model on test set"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    model = AudioDeepfakeDetector()
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Load test data
    # Load test data - custom 3000 files from bottom
    from torch.utils.data import DataLoader
    from audio_dataset import AudioDeepfakeDataset

    # Create custom dataset with files from bottom
    dataset = AudioDeepfakeDataset(data_dir, split='test')

    # Override the samples to get last 3000 files
    all_samples = []
    valid_dir = Path('data/raw/valid')  # Fixed path to valid folder
    for class_name in ['real', 'fake']:  # lowercase folder names
        class_dir = valid_dir / class_name
        audio_files = list(class_dir.glob('*.wav')) + list(class_dir.glob('*.mp3'))
        label = 0 if class_name == 'real' else 1
        all_samples.extend([(str(f), label) for f in audio_files])

    dataset.samples = all_samples
    print(f"Evaluating on {len(all_samples)} audio files from valid/raw/audio folders")

    test_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for spectrograms, labels in tqdm(test_loader, desc="Evaluating"):
            spectrograms = spectrograms.to(device)
            outputs = model(spectrograms)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    
    print("\n📊 Test Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"\nConfusion Matrix:\n{cm}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', default='checkpoints/best_model.pth')
    parser.add_argument('--data-dir', default='data/raw',
                       help='Directory containing REAL and FAKE folders')
    
    args = parser.parse_args()
    evaluate_model(args.model_path, args.data_dir)