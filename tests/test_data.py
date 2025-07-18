import unittest
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import tempfile
import shutil
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data.preprocessing import DeepfakePreprocessor
from src.data.dataset import DeepfakeDataset, create_data_loaders
from src.data.augmentation import AdvancedAugmentation, CompressionAugmentation

class TestDeepfakePreprocessor(unittest.TestCase):
    """Test deepfake preprocessor"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.preprocessor = DeepfakePreprocessor(
            input_size=224,
            normalize=True,
            augment=True
        )
        
        # Create test image
        self.test_image = Image.new('RGB', (512, 512), color='red')
        self.test_array = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    
    def test_preprocess_pil_image(self):
        """Test preprocessing PIL image"""
        processed = self.preprocessor.preprocess_image(self.test_image, augment=False)
        
        # Check output type and shape
        self.assertIsInstance(processed, torch.Tensor)
        self.assertEqual(processed.shape, (3, 224, 224))
        
        # Check value range (should be normalized)
        self.assertTrue(processed.min() >= -3)  # Roughly normalized range
        self.assertTrue(processed.max() <= 3)
    
    def test_preprocess_numpy_array(self):
        """Test preprocessing numpy array"""
        processed = self.preprocessor.preprocess_image(self.test_array, augment=False)
        
        self.assertIsInstance(processed, torch.Tensor)
        self.assertEqual(processed.shape, (3, 224, 224))
    
    def test_batch_preprocessing(self):
        """Test batch preprocessing"""
        images = [self.test_image, self.test_array]
        
        processed_batch = self.preprocessor.preprocess_batch(images, augment=False)
        
        self.assertIsInstance(processed_batch, torch.Tensor)
        self.assertEqual(processed_batch.shape, (2, 3, 224, 224))
    
    def test_denormalization(self):
        """Test denormalization"""
        processed = self.preprocessor.preprocess_image(self.test_image, augment=False)
        denormalized = self.preprocessor.denormalize(processed)
        
        # Check value range after denormalization
        self.assertTrue(denormalized.min() >= 0)
        self.assertTrue(denormalized.max() <= 1)
    
    def test_tensor_to_image(self):
        """Test converting tensor back to image"""
        processed = self.preprocessor.preprocess_image(self.test_image, augment=False)
        reconstructed = self.preprocessor.tensor_to_image(processed)
        
        self.assertIsInstance(reconstructed, Image.Image)
        self.assertEqual(reconstructed.size, (224, 224))

class TestAdvancedAugmentation(unittest.TestCase):
    """Test advanced augmentation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.augmentation = AdvancedAugmentation(
            input_size=224,
            augmentation_strength="medium"
        )
        
        # Create test image
        self.test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    
    def test_light_augmentation(self):
        """Test light augmentation"""
        aug_light = AdvancedAugmentation(augmentation_strength="light")
        
        augmented = aug_light(self.test_image, training=True)
        
        self.assertIsInstance(augmented, torch.Tensor)
        self.assertEqual(augmented.shape, (3, 224, 224))
    
    def test_medium_augmentation(self):
        """Test medium augmentation"""
        augmented = self.augmentation(self.test_image, training=True)
        
        self.assertIsInstance(augmented, torch.Tensor)
        self.assertEqual(augmented.shape, (3, 224, 224))
    
    def test_strong_augmentation(self):
        """Test strong augmentation"""
        aug_strong = AdvancedAugmentation(augmentation_strength="strong")
        
        augmented = aug_strong(self.test_image, training=True)
        
        self.assertIsInstance(augmented, torch.Tensor)
        self.assertEqual(augmented.shape, (3, 224, 224))
    
    def test_validation_augmentation(self):
        """Test validation (no augmentation)"""
        augmented = self.augmentation(self.test_image, training=False)
        
        self.assertIsInstance(augmented, torch.Tensor)
        self.assertEqual(augmented.shape, (3, 224, 224))

class TestCompressionAugmentation(unittest.TestCase):
    """Test compression augmentation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.compression_aug = CompressionAugmentation(quality_range=(50, 95), p=1.0)
        self.test_image = Image.new('RGB', (256, 256), color='blue')
    
    def test_compression_application(self):
        """Test that compression is applied"""
        # Apply compression
        compressed = self.compression_aug(self.test_image)
        
        self.assertIsInstance(compressed, np.ndarray)
        self.assertEqual(compressed.shape, (256, 256, 3))

class TestDeepfakeDataset(unittest.TestCase):
    """Test deepfake dataset"""
    
    def setUp(self):
        """Set up test fixtures with temporary dataset"""
        # Create temporary directory structure
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.temp_dir)
        
        # Create directory structure
        for split in ['train', 'val', 'test']:
            for class_name in ['real', 'fake']:
                (self.data_dir / 'processed' / split / class_name).mkdir(parents=True, exist_ok=True)
        
        # Create dummy images
        self._create_dummy_images()
        
        self.preprocessor = DeepfakePreprocessor(input_size=224)
    
    def tearDown(self):
        """Clean up temporary directory"""
        shutil.rmtree(self.temp_dir)
    
    def _create_dummy_images(self):
        """Create dummy images for testing"""
        for split in ['train', 'val', 'test']:
            for class_name in ['real', 'fake']:
                class_dir = self.data_dir / 'processed' / split / class_name
                
                # Create 5 dummy images per class
                for i in range(5):
                    img = Image.new('RGB', (256, 256), 
                                  color='red' if class_name == 'real' else 'blue')
                    img.save(class_dir / f'{class_name}_{i:03d}.jpg')
    
    def test_dataset_creation(self):
        """Test dataset creation"""
        dataset = DeepfakeDataset(
            data_dir=str(self.data_dir),
            split='train',
            preprocessor=self.preprocessor
        )
        
        self.assertEqual(len(dataset), 10)  # 5 real + 5 fake
        self.assertEqual(dataset.class_to_idx, {'real': 0, 'fake': 1})
    
    def test_dataset_getitem(self):
        """Test getting items from dataset"""
        dataset = DeepfakeDataset(
            data_dir=str(self.data_dir),
            split='train',
            preprocessor=self.preprocessor
        )
        
        image, label = dataset[0]
        
        self.assertIsInstance(image, torch.Tensor)
        self.assertEqual(image.shape, (3, 224, 224))
        self.assertIn(label, [0, 1])
    
    def test_class_balancing(self):
        """Test class balancing"""
        dataset = DeepfakeDataset(
            data_dir=str(self.data_dir),
            split='train',
            preprocessor=self.preprocessor,
            balance_classes=True
        )
        
        # Count labels
        labels = [dataset[i][1] for i in range(len(dataset))]
        real_count = labels.count(0)
        fake_count = labels.count(1)
        
        # Should be balanced
        self.assertEqual(real_count, fake_count)
    
    def test_class_weights(self):
        """Test class weights calculation"""
        dataset = DeepfakeDataset(
            data_dir=str(self.data_dir),
            split='train',
            preprocessor=self.preprocessor
        )
        
        weights = dataset.get_class_weights()
        
        self.assertIsInstance(weights, torch.Tensor)
        self.assertEqual(len(weights), 2)
        self.assertTrue(torch.all(weights > 0))

class TestDataLoaders(unittest.TestCase):
    """Test data loader creation"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create temporary directory structure
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.temp_dir)
        
        # Create directory structure
        for split in ['train', 'val', 'test']:
            for class_name in ['real', 'fake']:
                (self.data_dir / 'processed' / split / class_name).mkdir(parents=True, exist_ok=True)
        
        # Create dummy images
        for split in ['train', 'val', 'test']:
            for class_name in ['real', 'fake']:
                class_dir = self.data_dir / 'processed' / split / class_name
                
                for i in range(3):  # Fewer images for faster testing
                    img = Image.new('RGB', (128, 128), 
                                  color='red' if class_name == 'real' else 'blue')
                    img.save(class_dir / f'{class_name}_{i:03d}.jpg')
        
        self.preprocessor = DeepfakePreprocessor(input_size=128)
    
    def tearDown(self):
        """Clean up"""
        shutil.rmtree(self.temp_dir)
    
    def test_data_loader_creation(self):
        """Test creating data loaders"""
        train_loader, val_loader, test_loader = create_data_loaders(
            data_dir=str(self.data_dir),
            preprocessor=self.preprocessor,
            batch_size=2,
            num_workers=0  # No multiprocessing for testing
        )
        
        # Test that loaders are created
        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(val_loader)
        self.assertIsNotNone(test_loader)
        
        # Test batch from train loader
        batch = next(iter(train_loader))
        images, labels = batch
        
        self.assertEqual(images.shape[0], 2)  # batch size
        self.assertEqual(images.shape[1:], (3, 128, 128))  # image shape
        self.assertEqual(labels.shape[0], 2)  # batch size
    
    def test_data_loader_iteration(self):
        """Test iterating through data loader"""
        train_loader, _, _ = create_data_loaders(
            data_dir=str(self.data_dir),
            preprocessor=self.preprocessor,
            batch_size=2,
            num_workers=0
        )
        
        batch_count = 0
        for batch in train_loader:
            images, labels = batch
            
            # Check batch properties
            self.assertTrue(images.shape[0] <= 2)  # batch size or smaller for last batch
            self.assertEqual(images.shape[1:], (3, 128, 128))
            self.assertTrue(torch.all((labels == 0) | (labels == 1)))
            
            batch_count += 1
        
        self.assertTrue(batch_count > 0)

def run_data_tests():
    """Run all data tests"""
    test_suite = unittest.TestSuite()
    
    test_classes = [
        TestDeepfakePreprocessor,
        TestAdvancedAugmentation,
        TestCompressionAugmentation,
        TestDeepfakeDataset,
        TestDataLoaders
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result

if __name__ == "__main__":
    print("🧪 Running Data Tests...")
    print("=" * 50)
    
    result = run_data_tests()
    
    if result.wasSuccessful():
        print("\n✅ All data tests passed!")
    else:
        print(f"\n❌ {len(result.failures)} test(s) failed")
        print(f"❌ {len(result.errors)} error(s) occurred")
