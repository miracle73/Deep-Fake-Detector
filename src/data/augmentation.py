import torch
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2
from PIL import Image
from typing import Union, Optional, Tuple
import random

class AdvancedAugmentation:
    """Advanced augmentation pipeline for deepfake detection"""
    
    def __init__(
        self,
        input_size: int = 224,
        augmentation_strength: str = "medium",
        preserve_aspect_ratio: bool = True
    ):
        self.input_size = input_size
        self.augmentation_strength = augmentation_strength
        self.preserve_aspect_ratio = preserve_aspect_ratio
        
        # Define augmentation pipelines
        self.augmentation_pipeline = self._create_augmentation_pipeline()
        self.light_pipeline = self._create_light_pipeline()
        
    def _create_augmentation_pipeline(self) -> A.Compose:
        """Create comprehensive augmentation pipeline using Albumentations"""
        
        if self.augmentation_strength == "light":
            augmentations = [
                A.Resize(self.input_size, self.input_size),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.1,
                    contrast_limit=0.1,
                    p=0.3
                ),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ]
        
        elif self.augmentation_strength == "medium":
            augmentations = [
                A.Resize(int(self.input_size * 1.1), int(self.input_size * 1.1)),
                A.RandomCrop(self.input_size, self.input_size, p=1.0),
                A.HorizontalFlip(p=0.5),
                A.OneOf([
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2,
                        contrast_limit=0.2,
                        p=1.0
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=10,
                        sat_shift_limit=15,
                        val_shift_limit=10,
                        p=1.0
                    ),
                    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0)
                ], p=0.5),
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                    A.GaussianBlur(blur_limit=(1, 3), p=1.0),
                    A.MotionBlur(blur_limit=3, p=1.0)
                ], p=0.3),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.1,
                    rotate_limit=15,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=0.5
                ),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ]
        
        elif self.augmentation_strength == "strong":
            augmentations = [
                A.Resize(int(self.input_size * 1.2), int(self.input_size * 1.2)),
                A.RandomCrop(self.input_size, self.input_size, p=1.0),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.3,
                    contrast_limit=0.3,
                    p=0.7
                ),
                A.HueSaturationValue(
                    hue_shift_limit=20,
                    sat_shift_limit=30,
                    val_shift_limit=20,
                    p=0.5
                ),
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 80.0), p=1.0),
                    A.GaussianBlur(blur_limit=(1, 5), p=1.0),
                    A.MotionBlur(blur_limit=5, p=1.0),
                    A.MedianBlur(blur_limit=3, p=1.0)
                ], p=0.5),
                A.OneOf([
                    A.OpticalDistortion(distort_limit=0.1, shift_limit=0.1, p=1.0),
                    A.GridDistortion(num_steps=5, distort_limit=0.1, p=1.0),
                    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0)
                ], p=0.3),
                A.ShiftScaleRotate(
                    shift_limit=0.15,
                    scale_limit=0.15,
                    rotate_limit=25,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=0.7
                ),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
                A.RandomGamma(gamma_limit=(80, 120), p=0.3),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ]
        
        return A.Compose(augmentations)
    
    def _create_light_pipeline(self) -> A.Compose:
        """Create light augmentation for validation/test"""
        return A.Compose([
            A.Resize(self.input_size, self.input_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    
    def __call__(self, image: Union[np.ndarray, Image.Image], training: bool = True) -> torch.Tensor:
        """Apply augmentation to image"""
        
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Ensure RGB format
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Already RGB
            pass
        elif len(image.shape) == 3 and image.shape[2] == 4:
            # RGBA to RGB
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif len(image.shape) == 2:
            # Grayscale to RGB
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Apply appropriate pipeline
        if training:
            transformed = self.augmentation_pipeline(image=image)
        else:
            transformed = self.light_pipeline(image=image)
        
        return transformed['image']

class CompressionAugmentation:
    """JPEG compression augmentation to simulate real-world scenarios"""
    
    def __init__(self, quality_range: Tuple[int, int] = (70, 95), p: float = 0.3):
        self.quality_range = quality_range
        self.p = p
    
    def __call__(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """Apply JPEG compression"""
        if random.random() > self.p:
            return np.array(image) if isinstance(image, Image.Image) else image
        
        # Convert to PIL if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Apply JPEG compression
        import io
        quality = random.randint(*self.quality_range)
        
        # Save to bytes with compression
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        
        # Load back
        compressed_image = Image.open(buffer)
        return np.array(compressed_image)

class FaceAugmentation:
    """Specialized augmentation for face-focused deepfake detection"""
    
    def __init__(self, input_size: int = 224):
        self.input_size = input_size
        self.face_augmentation = A.Compose([
            A.Resize(int(input_size * 1.1), int(input_size * 1.1)),
            A.RandomCrop(input_size, input_size),
            A.HorizontalFlip(p=0.5),
            
            # Face-specific augmentations
            A.RandomBrightnessContrast(
                brightness_limit=0.15,
                contrast_limit=0.15,
                p=0.6
            ),
            
            # Skin tone variations
            A.HueSaturationValue(
                hue_shift_limit=5,
                sat_shift_limit=20,
                val_shift_limit=10,
                p=0.4
            ),
            
            # Subtle geometric transforms (faces are sensitive)
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.05,
                rotate_limit=10,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.4
            ),
            
            # Lighting effects
            A.RandomShadow(
                shadow_roi=(0, 0.5, 1, 1),
                num_shadows_lower=1,
                num_shadows_upper=2,
                shadow_dimension=5,
                p=0.3
            ),
            
            # Subtle noise (deepfakes often have compression artifacts)
            A.OneOf([
                A.GaussNoise(var_limit=(5.0, 25.0), p=1.0),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0)
            ], p=0.3),
            
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    
    def __call__(self, image: Union[np.ndarray, Image.Image]) -> torch.Tensor:
        """Apply face-specific augmentation"""
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        transformed = self.face_augmentation(image=image)
        return transformed['image']