import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
from typing import Tuple, Optional, Union

class DeepfakePreprocessor:
    """Preprocessing pipeline for deepfake detection"""
    
    def __init__(
        self,
        input_size: int = 224,
        normalize: bool = True,
        augment: bool = False
    ):
        self.input_size = input_size
        self.normalize = normalize
        self.augment = augment
        
        # ImageNet statistics for normalization
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        
        self.transforms = self._build_transforms()
    
    def _build_transforms(self):
        """Build transformation pipeline"""
        transform_list = []
        
        # Resize and convert to tensor
        transform_list.extend([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor()
        ])
        
        # Normalization
        if self.normalize:
            transform_list.append(
                transforms.Normalize(mean=self.mean, std=self.std)
            )
        
        return transforms.Compose(transform_list)
    
    def _build_augmentation_transforms(self):
        """Build augmentation pipeline for training"""
        if not self.augment:
            return self.transforms
        
        augment_list = [
            transforms.Resize((self.input_size + 32, self.input_size + 32)),
            transforms.RandomCrop((self.input_size, self.input_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor()
        ]
        
        if self.normalize:
            augment_list.append(
                transforms.Normalize(mean=self.mean, std=self.std)
            )
        
        return transforms.Compose(augment_list)
    
    def preprocess_image(
        self, 
        image: Union[str, Image.Image, np.ndarray],
        augment: bool = False
    ) -> torch.Tensor:
        """Preprocess a single image"""
        
        # Load image if path provided
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Choose transform based on augmentation
        if augment and self.augment:
            transform = self._build_augmentation_transforms()
        else:
            transform = self.transforms
        
        # Apply transforms
        processed = transform(image)
        
        return processed
    
    def preprocess_batch(
        self,
        images: list,
        augment: bool = False
    ) -> torch.Tensor:
        """Preprocess a batch of images"""
        processed_images = []
        
        for image in images:
            processed = self.preprocess_image(image, augment=augment)
            processed_images.append(processed)
        
        # Stack into batch
        batch = torch.stack(processed_images)
        return batch
    
    def denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Denormalize tensor for visualization"""
        if not self.normalize:
            return tensor
        
        # Create denormalization transform
        denorm = transforms.Normalize(
            mean=[-m/s for m, s in zip(self.mean, self.std)],
            std=[1/s for s in self.std]
        )
        
        return denorm(tensor)
    
    def tensor_to_image(self, tensor: torch.Tensor) -> Image.Image:
        """Convert tensor back to PIL Image"""
        # Denormalize if needed
        if self.normalize:
            tensor = self.denormalize(tensor)
        
        # Clamp values and convert to numpy
        tensor = torch.clamp(tensor, 0, 1)
        numpy_image = tensor.permute(1, 2, 0).numpy()
        numpy_image = (numpy_image * 255).astype(np.uint8)
        
        return Image.fromarray(numpy_image)

def enhance_image_quality(image: np.ndarray) -> np.ndarray:
    """Enhance image quality for better detection"""
    # Convert to LAB color space for better processing
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    
    # Merge channels and convert back to RGB
    enhanced_lab = cv2.merge([l, a, b])
    enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    
    return enhanced_rgb

def detect_and_extract_faces(image: np.ndarray) -> list:
    """Detect and extract faces from image (optional preprocessing step)"""
    try:
        import cv2
        
        # Load face cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Convert to grayscale for detection
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        face_crops = []
        for (x, y, w, h) in faces:
            # Add padding
            padding = int(0.2 * max(w, h))
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(image.shape[1], x + w + padding)
            y2 = min(image.shape[0], y + h + padding)
            
            face_crop = image[y1:y2, x1:x2]
            face_crops.append(face_crop)
        
        return face_crops
    
    except Exception as e:
        print(f"Face detection failed: {e}")
        return [image]  # Return original image if face detection fails