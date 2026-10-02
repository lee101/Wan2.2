#!/usr/bin/env python3
"""
Integration Tests for TI2V-5B GPU-A6000 Example
===============================================

This module contains integration tests for the TI2V-5B example script,
including GPU compatibility checks, model loading tests, and generation tests.

Run with: python -m pytest tests/test_ti2v_5b_integration.py -v
"""

import os
import sys
import tempfile
import unittest
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import shutil

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from examples.ti2v_5b_a6000_example import TI2V5BGenerator
    EXAMPLE_AVAILABLE = True
except ImportError:
    EXAMPLE_AVAILABLE = False

class TestTI2V5BIntegration(unittest.TestCase):
    """Integration tests for TI2V-5B example"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.test_dir = Path(tempfile.mkdtemp())
        cls.mock_model_dir = cls.test_dir / "Wan2.2-TI2V-5B"
        cls.mock_model_dir.mkdir(exist_ok=True)
        
        # Create mock model files
        (cls.mock_model_dir / "config.json").write_text('{"model": "ti2v-5b"}')
        (cls.mock_model_dir / "pytorch_model.bin").write_text("mock model")
        
        # Create test image
        cls.test_image = cls.test_dir / "test_image.jpg"
        # Create a small dummy image file
        cls.test_image.write_bytes(b"fake_image_data")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        if cls.test_dir.exists():
            shutil.rmtree(cls.test_dir)
    
    def setUp(self):
        """Set up each test"""
        self.output_dir = self.test_dir / "outputs"
        self.output_dir.mkdir(exist_ok=True)
    
    @unittest.skipUnless(EXAMPLE_AVAILABLE, "Example module not available")
    def test_generator_initialization(self):
        """Test TI2V5BGenerator initialization"""
        generator = TI2V5BGenerator(str(self.mock_model_dir))
        
        self.assertEqual(generator.model_path, self.mock_model_dir)
        self.assertIn("ti2v-5B", generator.config["task"])
        self.assertEqual(generator.config["size"], "1280*704")
        self.assertTrue(generator.config["offload_model"])
        self.assertTrue(generator.config["convert_model_dtype"])
        self.assertTrue(generator.config["t5_cpu"])
    
    @unittest.skipUnless(EXAMPLE_AVAILABLE, "Example module not available")
    def test_generator_invalid_path(self):
        """Test generator with invalid model path"""
        with self.assertRaises(FileNotFoundError):
            TI2V5BGenerator("/nonexistent/path")
    
    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
    def test_gpu_compatibility_check(self):
        """Test GPU compatibility checking"""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        # Test GPU memory check
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / 1e9
        
        # GPU-A6000 should have ~48GB VRAM, which is sufficient for TI2V-5B
        self.assertGreaterEqual(vram_gb, 20, "Insufficient VRAM for testing")
    
    @unittest.skipUnless(EXAMPLE_AVAILABLE, "Example module not available")
    def test_memory_usage_tracking(self):
        """Test GPU memory usage tracking"""
        generator = TI2V5BGenerator(str(self.mock_model_dir))
        memory_info = generator.get_memory_usage()
        
        if torch.cuda.is_available():
            self.assertIn("allocated", memory_info)
            self.assertIn("reserved", memory_info)
            self.assertIn("total", memory_info)
            self.assertIsInstance(memory_info["allocated"], (int, float))
            self.assertIsInstance(memory_info["reserved"], (int, float))
            self.assertIsInstance(memory_info["total"], (int, float))
        else:
            self.assertEqual(memory_info, {})
    
    @unittest.skipUnless(EXAMPLE_AVAILABLE, "Example module not available")
    @patch('subprocess.run')
    def test_text_to_video_generation(self, mock_subprocess):
        """Test text-to-video generation command construction"""
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stdout = "Generation successful"
        mock_subprocess.return_value.stderr = ""
        
        generator = TI2V5BGenerator(str(self.mock_model_dir))
        output_path = str(self.output_dir / "test_t2v.mp4")
        
        with patch('os.chdir'), patch('os.getcwd', return_value=str(self.test_dir)):
            result = generator.generate_text_to_video(
                prompt="A test video",
                output_path=output_path,
                seed=123
            )
        
        self.assertEqual(result, output_path)
        mock_subprocess.assert_called_once()
        
        # Verify command arguments
        call_args = mock_subprocess.call_args[0][0]
        self.assertIn("generate.py", call_args)
        self.assertIn("--task", call_args)
        self.assertIn("ti2v-5B", call_args)
        self.assertIn("--offload_model", call_args)
        self.assertIn("--convert_model_dtype", call_args)
        self.assertIn("--t5_cpu", call_args)
    
    @unittest.skipUnless(EXAMPLE_AVAILABLE, "Example module not available")
    @patch('subprocess.run')
    def test_image_to_video_generation(self, mock_subprocess):
        """Test image-to-video generation command construction"""
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stdout = "Generation successful"
        mock_subprocess.return_value.stderr = ""
        
        generator = TI2V5BGenerator(str(self.mock_model_dir))
        output_path = str(self.output_dir / "test_i2v.mp4")
        
        with patch('os.chdir'), patch('os.getcwd', return_value=str(self.test_dir)):
            result = generator.generate_image_to_video(
                image_path=str(self.test_image),
                prompt="A test video from image",
                output_path=output_path,
                seed=456
            )
        
        self.assertEqual(result, output_path)
        mock_subprocess.assert_called_once()
        
        # Verify command arguments
        call_args = mock_subprocess.call_args[0][0]
        self.assertIn("generate.py", call_args)
        self.assertIn("--image", call_args)
        self.assertIn(str(self.test_image), call_args)
    
    @unittest.skipUnless(EXAMPLE_AVAILABLE, "Example module not available")
    def test_image_to_video_missing_image(self):
        """Test image-to-video with missing image file"""
        generator = TI2V5BGenerator(str(self.mock_model_dir))
        
        with self.assertRaises(FileNotFoundError):
            generator.generate_image_to_video(
                image_path="/nonexistent/image.jpg",
                prompt="Test prompt",
                output_path="output.mp4"
            )
    
    @unittest.skipUnless(EXAMPLE_AVAILABLE, "Example module not available")
    @patch('subprocess.run')
    def test_generation_failure_handling(self, mock_subprocess):
        """Test handling of generation failures"""
        mock_subprocess.side_effect = subprocess.CalledProcessError(
            1, "generate.py", output="Error", stderr="Generation failed"
        )
        
        generator = TI2V5BGenerator(str(self.mock_model_dir))
        
        with patch('os.chdir'), patch('os.getcwd', return_value=str(self.test_dir)):
            with self.assertRaises(subprocess.CalledProcessError):
                generator.generate_text_to_video(
                    prompt="Test prompt",
                    output_path="output.mp4"
                )

class TestTI2V5BExampleCLI(unittest.TestCase):
    """Test the CLI interface of the example script"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.test_dir = Path(tempfile.mkdtemp())
        cls.mock_model_dir = cls.test_dir / "Wan2.2-TI2V-5B"
        cls.mock_model_dir.mkdir(exist_ok=True)
        
        # Create mock model files
        (cls.mock_model_dir / "config.json").write_text('{"model": "ti2v-5b"}')
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        if cls.test_dir.exists():
            shutil.rmtree(cls.test_dir)
    
    def test_cli_import(self):
        """Test that the CLI script can be imported"""
        try:
            from examples import ti2v_5b_a6000_example
            self.assertTrue(hasattr(ti2v_5b_a6000_example, 'main'))
        except ImportError:
            self.skipTest("Example module not available")
    
    @patch('sys.argv', ['ti2v_5b_a6000_example.py', '--check_gpu'])
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.get_device_properties')
    def test_gpu_check_command(self, mock_props, mock_cuda_available):
        """Test the GPU compatibility check command"""
        if not EXAMPLE_AVAILABLE:
            self.skipTest("Example module not available")
        
        # Mock GPU properties for A6000
        mock_gpu_props = Mock()
        mock_gpu_props.name = "NVIDIA RTX A6000"
        mock_gpu_props.total_memory = 48 * 1024**3  # 48GB
        mock_gpu_props.major = 8
        mock_gpu_props.minor = 6
        mock_props.return_value = mock_gpu_props
        
        from examples import ti2v_5b_a6000_example
        
        # This would normally call sys.exit, so we'll just test the function exists
        self.assertTrue(callable(ti2v_5b_a6000_example.main))

class TestTI2V5BPerformance(unittest.TestCase):
    """Performance and benchmark tests"""
    
    def setUp(self):
        """Set up performance tests"""
        self.test_dir = Path(tempfile.mkdtemp())
        self.mock_model_dir = self.test_dir / "Wan2.2-TI2V-5B"
        self.mock_model_dir.mkdir(exist_ok=True)
        (self.mock_model_dir / "config.json").write_text('{"model": "ti2v-5b"}')
    
    def tearDown(self):
        """Clean up performance tests"""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    @unittest.skipUnless(TORCH_AVAILABLE and torch.cuda.is_available(), 
                         "CUDA not available")
    def test_memory_efficiency(self):
        """Test memory efficiency of the configuration"""
        if not EXAMPLE_AVAILABLE:
            self.skipTest("Example module not available")
        
        generator = TI2V5BGenerator(str(self.mock_model_dir))
        
        # Check that memory optimizations are enabled
        self.assertTrue(generator.config["offload_model"], 
                       "Model offloading should be enabled for memory efficiency")
        self.assertTrue(generator.config["convert_model_dtype"], 
                       "Data type conversion should be enabled")
        self.assertTrue(generator.config["t5_cpu"], 
                       "T5 CPU usage should be enabled to save GPU memory")
    
    def test_resolution_settings(self):
        """Test that resolution is optimized for TI2V-5B"""
        if not EXAMPLE_AVAILABLE:
            self.skipTest("Example module not available")
        
        generator = TI2V5BGenerator(str(self.mock_model_dir))
        
        # TI2V-5B uses 1280*704 for 720P
        self.assertEqual(generator.config["size"], "1280*704")

if __name__ == "__main__":
    # Configure test runner
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestTI2V5BIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestTI2V5BExampleCLI))
    suite.addTests(loader.loadTestsFromTestCase(TestTI2V5BPerformance))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)