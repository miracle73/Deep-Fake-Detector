import unittest
import torch
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.efficientnet_detector import EfficientNetDeepfakeDetector, create_efficientnet_detector
from models.resnet_detector import ResNetDeepfakeDetector, create_resnet_detector
from models.ensemble_model import EnsembleDeepfakeDetector, create_efficientnet_resnet_ensemble

class TestEfficientNetDetector(unittest.TestCase):
    """Test EfficientNet deepfake detector"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.batch_size = 4
        self.input_size = 224
        self.num_classes = 2
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create test input
        self.test_input = torch.randn(self.batch_size, 3, self.input_size, self.input_size)
        self.test_input = self.test_input.to(self.device)
        
        # Create test labels
        self.test_labels = torch.randint(0, self.num_classes, (self.batch_size,))
        self.test_labels = self.test_labels.to(self.device)
    
    def test_model_creation(self):
        """Test model creation"""
        model = create_efficientnet_detector()
        self.assertIsInstance(model, EfficientNetDeepfakeDetector)
        self.assertEqual(model.num_classes, 2)
    
    def test_forward_pass(self):
        """Test forward pass"""
        model = create_efficientnet_detector().to(self.device)
        model.eval()
        
        with torch.no_grad():
            outputs = model(self.test_input)
        
        # Check output shape
        expected_shape = (self.batch_size, self.num_classes)
        self.assertEqual(outputs.shape, expected_shape)
        
        # Check output is finite
        self.assertTrue(torch.isfinite(outputs).all())
    
    def test_predict_proba(self):
        """Test probability prediction"""
        model = create_efficientnet_detector().to(self.device)
        model.eval()
        
        probabilities = model.predict_proba(self.test_input)
        
        # Check shape
        expected_shape = (self.batch_size, self.num_classes)
        self.assertEqual(probabilities.shape, expected_shape)
        
        # Check probabilities sum to 1
        prob_sums = torch.sum(probabilities, dim=1)
        torch.testing.assert_close(prob_sums, torch.ones_like(prob_sums), rtol=1e-5, atol=1e-5)
        
        # Check probabilities are in [0, 1]
        self.assertTrue((probabilities >= 0).all())
        self.assertTrue((probabilities <= 1).all())
    
    def test_predict(self):
        """Test binary prediction"""
        model = create_efficientnet_detector().to(self.device)
        model.eval()
        
        predictions, confidences = model.predict(self.test_input)
        
        # Check predictions are binary
        self.assertTrue(torch.all((predictions == 0) | (predictions == 1)))
        
        # Check confidences are in [0, 1]
        self.assertTrue((confidences >= 0).all())
        self.assertTrue((confidences <= 1).all())
    
    def test_feature_extraction(self):
        """Test feature vector extraction"""
        model = create_efficientnet_detector().to(self.device)
        model.eval()
        
        features = model.get_feature_vector(self.test_input)
        
        # Check shape (should be batch_size x feature_dim)
        self.assertEqual(features.shape[0], self.batch_size)
        self.assertTrue(features.shape[1] > 0)  # Feature dimension should be positive
        
        # Check features are finite
        self.assertTrue(torch.isfinite(features).all())

class TestResNetDetector(unittest.TestCase):
    """Test ResNet deepfake detector"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.batch_size = 4
        self.input_size = 224
        self.num_classes = 2
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.test_input = torch.randn(self.batch_size, 3, self.input_size, self.input_size)
        self.test_input = self.test_input.to(self.device)
    
    def test_model_creation(self):
        """Test ResNet model creation"""
        model = create_resnet_detector(model_name="resnet18")
        self.assertIsInstance(model, ResNetDeepfakeDetector)
        self.assertEqual(model.num_classes, 2)
    
    def test_different_resnet_architectures(self):
        """Test different ResNet architectures"""
        architectures = ["resnet18", "resnet34", "resnet50"]
        
        for arch in architectures:
            with self.subTest(architecture=arch):
                model = create_resnet_detector(model_name=arch).to(self.device)
                model.eval()
                
                with torch.no_grad():
                    outputs = model(self.test_input)
                
                expected_shape = (self.batch_size, self.num_classes)
                self.assertEqual(outputs.shape, expected_shape)
    
    def test_forward_pass(self):
        """Test ResNet forward pass"""
        model = create_resnet_detector().to(self.device)
        model.eval()
        
        with torch.no_grad():
            outputs = model(self.test_input)
        
        expected_shape = (self.batch_size, self.num_classes)
        self.assertEqual(outputs.shape, expected_shape)
        self.assertTrue(torch.isfinite(outputs).all())

class TestEnsembleModel(unittest.TestCase):
    """Test ensemble deepfake detector"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.batch_size = 4
        self.input_size = 224
        self.num_classes = 2
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.test_input = torch.randn(self.batch_size, 3, self.input_size, self.input_size)
        self.test_input = self.test_input.to(self.device)
        
        # Create individual models
        self.efficientnet = create_efficientnet_detector().to(self.device)
        self.resnet = create_resnet_detector(model_name="resnet18").to(self.device)
    
    def test_ensemble_creation(self):
        """Test ensemble model creation"""
        ensemble = EnsembleDeepfakeDetector(
            models=[self.efficientnet, self.resnet],
            ensemble_method="averaging"
        )
        
        self.assertEqual(ensemble.num_models, 2)
        self.assertEqual(ensemble.ensemble_method, "averaging")
    
    def test_averaging_ensemble(self):
        """Test averaging ensemble method"""
        ensemble = EnsembleDeepfakeDetector(
            models=[self.efficientnet, self.resnet],
            ensemble_method="averaging"
        ).to(self.device)
        ensemble.eval()
        
        with torch.no_grad():
            outputs = ensemble(self.test_input)
        
        expected_shape = (self.batch_size, self.num_classes)
        self.assertEqual(outputs.shape, expected_shape)
        self.assertTrue(torch.isfinite(outputs).all())
    
    def test_voting_ensemble(self):
        """Test voting ensemble method"""
        ensemble = EnsembleDeepfakeDetector(
            models=[self.efficientnet, self.resnet],
            ensemble_method="voting"
        ).to(self.device)
        ensemble.eval()
        
        with torch.no_grad():
            outputs = ensemble(self.test_input)
        
        expected_shape = (self.batch_size, self.num_classes)
        self.assertEqual(outputs.shape, expected_shape)
    
    def test_learned_ensemble(self):
        """Test learned ensemble method"""
        ensemble = EnsembleDeepfakeDetector(
            models=[self.efficientnet, self.resnet],
            ensemble_method="learned"
        ).to(self.device)
        ensemble.eval()
        
        with torch.no_grad():
            outputs = ensemble(self.test_input)
        
        expected_shape = (self.batch_size, self.num_classes)
        self.assertEqual(outputs.shape, expected_shape)
        self.assertTrue(torch.isfinite(outputs).all())
    
    def test_individual_predictions(self):
        """Test getting individual model predictions"""
        ensemble = EnsembleDeepfakeDetector(
            models=[self.efficientnet, self.resnet],
            ensemble_method="averaging"
        ).to(self.device)
        ensemble.eval()
        
        individual_preds = ensemble.get_individual_predictions(self.test_input)
        
        self.assertEqual(len(individual_preds), 2)
        self.assertIn('model_0', individual_preds)
        self.assertIn('model_1', individual_preds)
        
        for pred in individual_preds.values():
            expected_shape = (self.batch_size, self.num_classes)
            self.assertEqual(pred.shape, expected_shape)
    
    def test_confidence_analysis(self):
        """Test prediction confidence analysis"""
        ensemble = EnsembleDeepfakeDetector(
            models=[self.efficientnet, self.resnet],
            ensemble_method="averaging"
        ).to(self.device)
        ensemble.eval()
        
        analysis = ensemble.get_prediction_confidence_analysis(self.test_input)
        
        required_keys = ['ensemble_prediction', 'individual_predictions', 
                        'prediction_agreement', 'average_confidence', 'ensemble_confidence']
        
        for key in required_keys:
            self.assertIn(key, analysis)
        
        # Check shapes
        self.assertEqual(analysis['ensemble_prediction'].shape, (self.batch_size, self.num_classes))
        self.assertEqual(analysis['prediction_agreement'].shape, (self.batch_size,))
        self.assertEqual(analysis['average_confidence'].shape, (self.batch_size,))

class TestModelIntegration(unittest.TestCase):
    """Integration tests for model compatibility"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.batch_size = 2
        self.input_size = 224
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.test_input = torch.randn(self.batch_size, 3, self.input_size, self.input_size)
        self.test_input = self.test_input.to(self.device)
    
    def test_model_compatibility(self):
        """Test that all models produce compatible outputs"""
        models = [
            create_efficientnet_detector(),
            create_resnet_detector(model_name="resnet18"),
            create_resnet_detector(model_name="resnet50")
        ]
        
        outputs = []
        for model in models:
            model = model.to(self.device)
            model.eval()
            
            with torch.no_grad():
                output = model(self.test_input)
                outputs.append(output)
        
        # All outputs should have the same shape
        reference_shape = outputs[0].shape
        for output in outputs[1:]:
            self.assertEqual(output.shape, reference_shape)
    
    def test_ensemble_with_different_models(self):
        """Test ensemble with different model combinations"""
        efficientnet = create_efficientnet_detector()
        resnet18 = create_resnet_detector(model_name="resnet18")
        resnet50 = create_resnet_detector(model_name="resnet50")
        
        # Test different ensemble combinations
        model_combinations = [
            [efficientnet, resnet18],
            [efficientnet, resnet50],
            [resnet18, resnet50],
            [efficientnet, resnet18, resnet50]
        ]
        
        for models in model_combinations:
            with self.subTest(num_models=len(models)):
                ensemble = EnsembleDeepfakeDetector(
                    models=models,
                    ensemble_method="averaging"
                ).to(self.device)
                ensemble.eval()
                
                with torch.no_grad():
                    outputs = ensemble(self.test_input)
                
                expected_shape = (self.batch_size, 2)
                self.assertEqual(outputs.shape, expected_shape)

def run_tests():
    """Run all model tests"""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestEfficientNetDetector,
        TestResNetDetector,
        TestEnsembleModel,
        TestModelIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result

if __name__ == "__main__":
    print("🧪 Running Model Tests...")
    print("=" * 50)
    
    result = run_tests()
    
    if result.wasSuccessful():
        print("\n✅ All tests passed!")
    else:
        print(f"\n❌ {len(result.failures)} test(s) failed")
        print(f"❌ {len(result.errors)} error(s) occurred")
