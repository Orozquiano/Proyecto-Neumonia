import csv
import os
import tempfile
import numpy as np
import pytest
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import cv2


# ============================================================================
# Test 1: Preprocessor.preprocess
# ============================================================================
class TestPreprocessor:
    """Tests for Preprocessor.preprocess method."""

    def test_preprocess_resizes_to_512x512(self):
        """Verifies the image is resized to 512x512."""
        from image.preprocessor import Preprocessor
        
        preprocessor = Preprocessor()
        # Create a BGR image of different size
        input_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        result = preprocessor.preprocess(input_array)
        
        # Shape should be (1, 512, 512, 1)
        assert result.shape[1] == 512
        assert result.shape[2] == 512

    def test_preprocess_converts_to_grayscale(self):
        """Verifies the image is converted to grayscale (single channel)."""
        from image.preprocessor import Preprocessor
        
        preprocessor = Preprocessor()
        # Create a BGR image
        input_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        result = preprocessor.preprocess(input_array)
        
        # Last dimension should be 1 (grayscale)
        assert result.shape[-1] == 1

    def test_preprocess_applies_clahe(self):
        """Verifies CLAHE is applied by checking the preprocessor's clahe attribute."""
        from image.preprocessor import Preprocessor
        
        preprocessor = Preprocessor()
        input_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        preprocessor.preprocess(input_array)
        
        # Verify CLAHE object was created
        assert preprocessor.clahe is not None

    def test_preprocess_normalizes_values(self):
        """Verifies pixel values are normalized to [0, 1] range."""
        from image.preprocessor import Preprocessor
        
        preprocessor = Preprocessor()
        input_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        result = preprocessor.preprocess(input_array)
        
        # Values should be in [0, 1] range after normalization
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_preprocess_expands_dimensions(self):
        """Verifies dimensions are expanded for batch and channel."""
        from image.preprocessor import Preprocessor
        
        preprocessor = Preprocessor()
        input_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        result = preprocessor.preprocess(input_array)
        
        # Result should be 4D: (batch, height, width, channels)
        assert len(result.shape) == 4
        assert result.shape[0] == 1  # batch dimension
        assert result.shape[-1] == 1  # channel dimension

    def test_preprocess_returns_correct_shape(self):
        """Verifies the complete output shape is (1, 512, 512, 1)."""
        from image.preprocessor import Preprocessor
        
        preprocessor = Preprocessor()
        input_array = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
        
        result = preprocessor.preprocess(input_array)
        
        assert result.shape == (1, 512, 512, 1)


# ============================================================================
# Test 2: ImageReader.read_dicom and ImageReader.read_image
# ============================================================================
class TestImageReader:
    """Tests for ImageReader.read_dicom and ImageReader.read_image methods."""

    def test_read_dicom_returns_rgb_array_and_pil_image(self):
        """Verifies read_dicom returns an RGB numpy array and a PIL Image."""
        from image.reader import ImageReader
        
        # Use sample DICOM file
        dicom_path = "samples/DICOM/normal (2).dcm"
        if not os.path.exists(dicom_path):
            pytest.skip("Sample DICOM file not found")
        
        img_rgb, img2show = ImageReader.read_dicom(dicom_path)
        
        # Check that img_rgb is a numpy array with 3 channels (RGB)
        assert isinstance(img_rgb, np.ndarray)
        assert len(img_rgb.shape) == 3
        assert img_rgb.shape[2] == 3  # RGB channels
        
        # Check that img2show is a PIL Image
        assert isinstance(img2show, Image.Image)

    def test_read_dicom_normalizes_pixel_values(self):
        """Verifies read_dicom normalizes pixel values to uint8 range."""
        from image.reader import ImageReader
        
        dicom_path = "samples/DICOM/normal (2).dcm"
        if not os.path.exists(dicom_path):
            pytest.skip("Sample DICOM file not found")
        
        img_rgb, _ = ImageReader.read_dicom(dicom_path)
        
        assert img_rgb.dtype == np.uint8
        assert img_rgb.max() <= 255
        assert img_rgb.min() >= 0

    def test_read_image_returns_array_and_pil_image(self):
        """Verifies read_image returns a numpy array and a PIL Image."""
        from image.reader import ImageReader
        
        # Use sample JPG file
        jpg_path = "samples/JPG/normal/NORMAL2-IM-1144-0001.jpeg"
        if not os.path.exists(jpg_path):
            pytest.skip("Sample JPG file not found")
        
        img_array, img2show = ImageReader.read_image(jpg_path)
        
        # Check that img_array is a numpy array
        assert isinstance(img_array, np.ndarray)
        
        # Check that img2show is a PIL Image
        assert isinstance(img2show, Image.Image)

    def test_read_image_normalizes_pixel_values(self):
        """Verifies read_image normalizes pixel values to uint8 range."""
        from image.reader import ImageReader
        
        jpg_path = "samples/JPG/normal/NORMAL2-IM-1144-0001.jpeg"
        if not os.path.exists(jpg_path):
            pytest.skip("Sample JPG file not found")
        
        img_array, _ = ImageReader.read_image(jpg_path)
        
        assert img_array.dtype == np.uint8
        assert img_array.max() <= 255
        assert img_array.min() >= 0

    def test_read_image_preserves_image_content(self):
        """Verifies read_image preserves the image content."""
        from image.reader import ImageReader
        
        jpg_path = "samples/JPG/bacteria/person1710_bacteria_4526.jpeg"
        if not os.path.exists(jpg_path):
            pytest.skip("Sample JPG file not found")
        
        img_array, img2show = ImageReader.read_image(jpg_path)
        
        # Image should have non-zero content
        assert img_array.size > 0
        assert img2show.size[0] > 0 and img2show.size[1] > 0


# ============================================================================
# Test 3: Predictor.predict
# ============================================================================
class TestPredictor:
    """Tests for Predictor.predict method."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock TensorFlow model."""
        model = Mock()
        # Mock model call to return predictions (3 classes)
        model.return_value = Mock()
        model.return_value.numpy = Mock(return_value=np.array([[0.1, 0.8, 0.1]]))
        model.inputs = [Mock()]
        model.output = Mock()
        model.layers = []
        return model

    def test_predict_returns_label_probability_heatmap(self, mock_model):
        """Verifies predict returns label, probability, and heatmap."""
        from model.predictor import Predictor
        
        # Create a mock conv layer for GradCAM
        mock_conv_layer = Mock(spec=['output'])
        mock_conv_layer.output = Mock()
        mock_model.layers = [mock_conv_layer]
        
        with patch('model.gradcam.tf.keras.layers.Conv2D', return_value=Mock()):
            with patch.object(mock_conv_layer, '__class__', new=type('Conv2D', (), {})):
                with patch('model.gradcam.tf.keras.models.Model') as mock_grad_model:
                    # Setup GradCAM mock
                    mock_grad_model_instance = Mock()
                    mock_grad_model.return_value = mock_grad_model_instance
                    mock_grad_model_instance.return_value = (
                        np.random.rand(1, 16, 16, 32).astype(np.float32),
                        np.array([[0.1, 0.8, 0.1]])
                    )
                    
                    with patch('model.gradcam.tf.GradientTape') as mock_tape:
                        mock_tape_instance = Mock()
                        mock_tape.return_value.__enter__ = Mock(return_value=mock_tape_instance)
                        mock_tape.return_value.__exit__ = Mock(return_value=False)
                        mock_tape_instance.gradient = Mock(return_value=np.random.rand(1, 16, 16, 32))
                        
                        with patch('model.gradcam.tf.reduce_mean', return_value=np.random.rand(32)):
                            with patch('model.gradcam.tf.reduce_sum', return_value=np.random.rand(16, 16)):
                                with patch('model.gradcam.tf.maximum', return_value=np.random.rand(16, 16)):
                                    with patch('model.gradcam.tf.reduce_max', return_value=1.0):
                                        predictor = Predictor(mock_model)
                                        
                                        # Create test input
                                        input_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
                                        
                                        with patch.object(predictor.gradcam, 'generate', return_value=np.zeros((512, 512, 3), dtype=np.uint8)):
                                            label, proba, heatmap = predictor.predict(input_array)
        
        # Verify return types
        assert isinstance(label, str)
        assert isinstance(proba, (int, float, np.floating))
        assert isinstance(heatmap, np.ndarray)

    def test_predict_returns_valid_label(self, mock_model):
        """Verifies predict returns one of the valid labels."""
        from model.predictor import Predictor
        
        with patch.object(Predictor, '__init__', lambda self, model: None):
            predictor = Predictor.__new__(Predictor)
            predictor.model = mock_model
            predictor.preprocessor = Mock()
            predictor.preprocessor.preprocess = Mock(return_value=np.zeros((1, 512, 512, 1)))
            predictor.gradcam = Mock()
            predictor.gradcam.generate = Mock(return_value=np.zeros((512, 512, 3), dtype=np.uint8))
            
            input_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            label, proba, heatmap = predictor.predict(input_array)
        
        assert label in ["bacteriana", "normal", "viral"]

    def test_predict_returns_probability_percentage(self, mock_model):
        """Verifies predict returns probability as percentage (0-100)."""
        from model.predictor import Predictor
        
        with patch.object(Predictor, '__init__', lambda self, model: None):
            predictor = Predictor.__new__(Predictor)
            predictor.model = mock_model
            predictor.preprocessor = Mock()
            predictor.preprocessor.preprocess = Mock(return_value=np.zeros((1, 512, 512, 1)))
            predictor.gradcam = Mock()
            predictor.gradcam.generate = Mock(return_value=np.zeros((512, 512, 3), dtype=np.uint8))
            
            input_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            label, proba, heatmap = predictor.predict(input_array)
        
        # Probability should be in percentage (0-100 range)
        assert 0 <= proba <= 100

    def test_predict_returns_heatmap_array(self, mock_model):
        """Verifies predict returns a valid heatmap numpy array."""
        from model.predictor import Predictor
        
        expected_heatmap = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        
        with patch.object(Predictor, '__init__', lambda self, model: None):
            predictor = Predictor.__new__(Predictor)
            predictor.model = mock_model
            predictor.preprocessor = Mock()
            predictor.preprocessor.preprocess = Mock(return_value=np.zeros((1, 512, 512, 1)))
            predictor.gradcam = Mock()
            predictor.gradcam.generate = Mock(return_value=expected_heatmap)
            
            input_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            label, proba, heatmap = predictor.predict(input_array)
        
        assert isinstance(heatmap, np.ndarray)
        np.testing.assert_array_equal(heatmap, expected_heatmap)


# ============================================================================
# Test 4: GradCAM.generate
# ============================================================================
class TestGradCAM:
    """Tests for GradCAM.generate method."""

    @pytest.fixture
    def mock_model_with_conv(self):
        """Create a mock model with a Conv2D layer."""
        import tensorflow as tf
        
        model = Mock()
        
        # Create a mock Conv2D layer
        conv_layer = Mock(spec=tf.keras.layers.Conv2D)
        conv_layer.output = Mock()
        
        model.layers = [conv_layer]
        model.inputs = [Mock()]
        model.output = Mock()
        
        return model, conv_layer

    def test_generate_returns_valid_heatmap_dimensions(self):
        """Verifies generate returns heatmap with correct dimensions (512x512x3)."""
        from model.gradcam import GradCAM
        import tensorflow as tf
        
        # Create mock model
        model = Mock()
        conv_layer = Mock(spec=tf.keras.layers.Conv2D)
        conv_layer.output = Mock()
        model.layers = [conv_layer]
        model.inputs = [Mock()]
        model.output = Mock()
        
        with patch('model.gradcam.tf.keras.models.Model') as mock_grad_model:
            mock_grad_model_instance = Mock()
            mock_grad_model.return_value = mock_grad_model_instance
            
            # Mock the forward pass
            conv_output = tf.constant(np.random.rand(1, 16, 16, 32).astype(np.float32))
            predictions = tf.constant(np.array([[0.1, 0.8, 0.1]]).astype(np.float32))
            mock_grad_model_instance.return_value = (conv_output, predictions)
            
            gradcam = GradCAM(model)
            
            img_array = np.zeros((1, 512, 512, 1), dtype=np.float32)
            original_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            
            heatmap = gradcam.generate(img_array, original_image)
        
        # Heatmap should be 512x512 with 3 channels (RGB)
        assert heatmap.shape == (512, 512, 3)

    def test_generate_returns_uint8_array(self):
        """Verifies generate returns uint8 numpy array."""
        from model.gradcam import GradCAM
        import tensorflow as tf
        
        model = Mock()
        conv_layer = Mock(spec=tf.keras.layers.Conv2D)
        conv_layer.output = Mock()
        model.layers = [conv_layer]
        model.inputs = [Mock()]
        model.output = Mock()
        
        with patch('model.gradcam.tf.keras.models.Model') as mock_grad_model:
            mock_grad_model_instance = Mock()
            mock_grad_model.return_value = mock_grad_model_instance
            
            conv_output = tf.constant(np.random.rand(1, 16, 16, 32).astype(np.float32))
            predictions = tf.constant(np.array([[0.1, 0.8, 0.1]]).astype(np.float32))
            mock_grad_model_instance.return_value = (conv_output, predictions)
            
            gradcam = GradCAM(model)
            
            img_array = np.zeros((1, 512, 512, 1), dtype=np.float32)
            original_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            
            heatmap = gradcam.generate(img_array, original_image)
        
        assert heatmap.dtype == np.uint8

    def test_generate_returns_valid_pixel_values(self):
        """Verifies generate returns heatmap with valid pixel values (0-255)."""
        from model.gradcam import GradCAM
        import tensorflow as tf
        
        model = Mock()
        conv_layer = Mock(spec=tf.keras.layers.Conv2D)
        conv_layer.output = Mock()
        model.layers = [conv_layer]
        model.inputs = [Mock()]
        model.output = Mock()
        
        with patch('model.gradcam.tf.keras.models.Model') as mock_grad_model:
            mock_grad_model_instance = Mock()
            mock_grad_model.return_value = mock_grad_model_instance
            
            conv_output = tf.constant(np.random.rand(1, 16, 16, 32).astype(np.float32))
            predictions = tf.constant(np.array([[0.1, 0.8, 0.1]]).astype(np.float32))
            mock_grad_model_instance.return_value = (conv_output, predictions)
            
            gradcam = GradCAM(model)
            
            img_array = np.zeros((1, 512, 512, 1), dtype=np.float32)
            original_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            
            heatmap = gradcam.generate(img_array, original_image)
        
        assert heatmap.min() >= 0
        assert heatmap.max() <= 255

    def test_get_last_conv_layer_finds_conv2d(self):
        """Verifies get_last_conv_layer correctly identifies Conv2D layer."""
        from model.gradcam import GradCAM
        import tensorflow as tf
        
        model = Mock()
        
        # Create layers with the last one being Conv2D
        dense_layer = Mock()
        conv_layer = Mock(spec=tf.keras.layers.Conv2D)
        
        model.layers = [conv_layer, dense_layer]
        
        gradcam = GradCAM(model)
        
        # Patch isinstance to work with our mock
        with patch('model.gradcam.isinstance', side_effect=lambda obj, cls: obj == conv_layer and cls == tf.keras.layers.Conv2D):
            result = gradcam.get_last_conv_layer()
        
        # Should return the conv_layer (last Conv2D in reversed order)
        assert result == conv_layer

    def test_get_last_conv_layer_raises_error_without_conv(self):
        """Verifies get_last_conv_layer raises error when no Conv2D layer exists."""
        from model.gradcam import GradCAM
        
        model = Mock()
        model.layers = [Mock(), Mock()]  # No Conv2D layers
        
        gradcam = GradCAM(model)
        
        with pytest.raises(ValueError, match="El modelo no contiene capas convolucionales"):
            gradcam.get_last_conv_layer()


# ============================================================================
# Test 5: HistoryService.save
# ============================================================================
class TestHistoryService:
    """Tests for HistoryService.save method."""

    def test_save_appends_to_csv_file(self):
        """Verifies save appends patient data to the CSV file."""
        from services.history_service import HistoryService
        
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "historial.csv")
            
            with patch('services.history_service.open', create=True) as mock_open:
                with patch('services.history_service.showinfo'):
                    mock_file = MagicMock()
                    mock_open.return_value.__enter__.return_value = mock_file
                    
                    with patch('csv.writer') as mock_writer:
                        mock_csv_writer = Mock()
                        mock_writer.return_value = mock_csv_writer
                        
                        HistoryService.save("P001", "normal", 95.5)
                        
                        # Verify file was opened in append mode
                        mock_open.assert_called_once_with("historial.csv", "a")
                        
                        # Verify row was written
                        mock_csv_writer.writerow.assert_called_once()

    def test_save_writes_correct_data_format(self):
        """Verifies save writes patient_id, label, and formatted probability."""
        from services.history_service import HistoryService
        
        with patch('services.history_service.open', create=True) as mock_open:
            with patch('services.history_service.showinfo'):
                mock_file = MagicMock()
                mock_open.return_value.__enter__.return_value = mock_file
                
                with patch('csv.writer') as mock_writer:
                    mock_csv_writer = Mock()
                    mock_writer.return_value = mock_csv_writer
                    
                    HistoryService.save("P123", "bacteriana", 87.65)
                    
                    # Verify the exact data written
                    mock_csv_writer.writerow.assert_called_with(
                        ["P123", "bacteriana", "87.65%"]
                    )

    def test_save_uses_dash_delimiter(self):
        """Verifies save uses dash (-) as CSV delimiter."""
        from services.history_service import HistoryService
        
        with patch('services.history_service.open', create=True) as mock_open:
            with patch('services.history_service.showinfo'):
                mock_file = MagicMock()
                mock_open.return_value.__enter__.return_value = mock_file
                
                with patch('csv.writer') as mock_writer:
                    mock_csv_writer = Mock()
                    mock_writer.return_value = mock_csv_writer
                    
                    HistoryService.save("P001", "viral", 75.0)
                    
                    # Verify delimiter was set to '-'
                    mock_writer.assert_called_with(mock_file, delimiter="-")

    def test_save_shows_success_message(self):
        """Verifies save displays a success message via showinfo."""
        from services.history_service import HistoryService
        
        with patch('services.history_service.open', create=True) as mock_open:
            mock_file = MagicMock()
            mock_open.return_value.__enter__.return_value = mock_file
            
            with patch('services.history_service.showinfo') as mock_showinfo:
                with patch('csv.writer') as mock_writer:
                    mock_writer.return_value = Mock()
                    
                    HistoryService.save("P001", "normal", 90.0)
                    
                    # Verify showinfo was called with success message
                    mock_showinfo.assert_called_once_with(
                        title="Guardar",
                        message="Los datos se guardaron con éxito."
                    )

    def test_save_formats_probability_with_two_decimals(self):
        """Verifies probability is formatted with exactly 2 decimal places."""
        from services.history_service import HistoryService
        
        with patch('services.history_service.open', create=True) as mock_open:
            with patch('services.history_service.showinfo'):
                mock_file = MagicMock()
                mock_open.return_value.__enter__.return_value = mock_file
                
                with patch('csv.writer') as mock_writer:
                    mock_csv_writer = Mock()
                    mock_writer.return_value = mock_csv_writer
                    
                    # Test with many decimal places
                    HistoryService.save("P001", "normal", 95.12345)
                    
                    # Should be formatted to 2 decimal places
                    call_args = mock_csv_writer.writerow.call_args[0][0]
                    assert call_args[2] == "95.12%"


# ============================================================================
# Integration-style tests (optional, require actual model)
# ============================================================================
class TestIntegration:
    """Integration tests that require actual files and model."""

    @pytest.mark.skipif(
        not os.path.exists("model/conv_MLP_84.h5"),
        reason="Model file not found"
    )
    def test_full_prediction_pipeline(self):
        """Tests the full prediction pipeline with actual model."""
        from model.model_loader import ModelLoader
        from model.predictor import Predictor
        from image.reader import ImageReader
        
        jpg_path = "samples/JPG/normal/NORMAL2-IM-1144-0001.jpeg"
        if not os.path.exists(jpg_path):
            pytest.skip("Sample image not found")
        
        # Load model
        model = ModelLoader.get_model()
        predictor = Predictor(model)
        
        # Read image
        img_array, _ = ImageReader.read_image(jpg_path)
        
        # Predict
        label, proba, heatmap = predictor.predict(img_array)
        
        # Verify outputs
        assert label in ["bacteriana", "normal", "viral"]
        assert 0 <= proba <= 100
        assert heatmap.shape == (512, 512, 3)
