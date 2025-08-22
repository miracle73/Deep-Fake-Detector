import numpy as np
import librosa
import torch
from pathlib import Path
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class AudioPreprocessor:
    """Preprocess audio files for deepfake detection"""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        max_duration: float = 10.0,
        augment: bool = False
    ):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.max_duration = max_duration
        self.augment = augment
        
        self.max_samples = int(max_duration * sample_rate)
    
    def load_audio(self, audio_path: str) -> Optional[np.ndarray]:
        """Load and resample audio file"""
        try:
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Trim or pad to max_duration
            if len(audio) > self.max_samples:
                audio = audio[:self.max_samples]
            elif len(audio) < self.max_samples:
                audio = np.pad(audio, (0, self.max_samples - len(audio)))
            
            return audio
            
        except Exception as e:
            logger.error(f"Error loading audio {audio_path}: {e}")
            return None
    
    def extract_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Extract mel-spectrogram features"""
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        # Convert to log scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize
        mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)
        
        return mel_spec_db
    
    def augment_audio(self, audio: np.ndarray) -> np.ndarray:
        """Apply augmentations to audio"""
        if not self.augment:
            return audio
        
        # Random noise
        if np.random.random() > 0.5:
            noise = np.random.normal(0, 0.005, audio.shape)
            audio = audio + noise
        
        # Random pitch shift
        if np.random.random() > 0.5:
            n_steps = np.random.randint(-2, 3)
            audio = librosa.effects.pitch_shift(audio, sr=self.sample_rate, n_steps=n_steps)
        
        # Random time stretch
        if np.random.random() > 0.5:
            rate = np.random.uniform(0.9, 1.1)
            audio = librosa.effects.time_stretch(audio, rate=rate)
            
            # Adjust length after stretching
            if len(audio) > self.max_samples:
                audio = audio[:self.max_samples]
            else:
                audio = np.pad(audio, (0, self.max_samples - len(audio)))
        
        return audio
    
    def process_audio_file(self, audio_path: str) -> Optional[np.ndarray]:
        """Complete preprocessing pipeline"""
        # Load audio
        audio = self.load_audio(audio_path)
        if audio is None:
            return None
        
        # Apply augmentations
        if self.augment:
            audio = self.augment_audio(audio)
        
        # Extract mel-spectrogram
        mel_spec = self.extract_mel_spectrogram(audio)
        
        return mel_spec