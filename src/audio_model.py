import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
import numpy as np

class AudioDeepfakeDetector(nn.Module):
    """Audio deepfake detection model using spectrograms and LSTM"""
    
    def __init__(
        self,
        input_size: int = 128,  # Mel-spectrogram frequency bins
        hidden_size: int = 256,
        num_layers: int = 3,
        num_classes: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        
        # CNN for spectrogram feature extraction
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # Keep time dimension
        )
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # x shape: (batch, 1, freq_bins, time_frames)
        
        # CNN feature extraction
        conv_out = self.conv_layers(x)  # (batch, 128, freq', time')
        
        # Reshape for LSTM: (batch, time, features)
        batch_size = conv_out.size(0)
        conv_out = conv_out.view(batch_size, 128, -1).transpose(1, 2)  # (batch, time, 128)
        
        # LSTM processing
        lstm_out, _ = self.lstm(conv_out)  # (batch, time, hidden*2)
        
        # Use last timestep for classification
        last_output = lstm_out[:, -1, :]
        
        # Classification
        logits = self.classifier(last_output)
        
        return logits
    
    def predict(self, spectrogram: np.ndarray) -> Dict[str, Any]:
        """Predict on a single audio spectrogram"""
        self.eval()
        
        with torch.no_grad():
            if isinstance(spectrogram, np.ndarray):
                spec_tensor = torch.from_numpy(spectrogram).float()
                if len(spec_tensor.shape) == 2:
                    spec_tensor = spec_tensor.unsqueeze(0).unsqueeze(0)
                elif len(spec_tensor.shape) == 3:
                    spec_tensor = spec_tensor.unsqueeze(0)
            
            logits = self.forward(spec_tensor)
            probabilities = F.softmax(logits, dim=1)
            
            fake_prob = probabilities[0, 1].item()
            real_prob = probabilities[0, 0].item()
            is_fake = fake_prob > 0.5
            
            return {
                'is_fake': is_fake,
                'fake_probability': fake_prob,
                'real_probability': real_prob,
                'confidence': max(fake_prob, real_prob)
            }