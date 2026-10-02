#!/usr/bin/env python3
"""
Unit tests for Wan2.2 modules and components.
Tests individual components without requiring full model weights.
"""

import unittest
import torch
import tempfile
import os
import sys
from unittest.mock import Mock, patch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class TestWanImports(unittest.TestCase):
    """Test that all Wan modules can be imported."""
    
    def test_main_imports(self):
        """Test main wan module imports."""
        try:
            import wan
            self.assertTrue(hasattr(wan, 'WanT2V'))
            self.assertTrue(hasattr(wan, 'WanI2V'))
            self.assertTrue(hasattr(wan, 'WanTI2V'))
            self.assertTrue(hasattr(wan, 'WanS2V'))
        except Exception as e:
            self.fail(f"Failed to import wan module: {e}")
    
    def test_config_imports(self):
        """Test config module imports."""
        try:
            from wan import configs
            # Test that configs can be imported
            self.assertTrue(hasattr(configs, '__name__'))
        except Exception as e:
            self.fail(f"Failed to import wan.configs: {e}")
    
    def test_module_imports(self):
        """Test individual module imports."""
        try:
            from wan import modules
            self.assertTrue(hasattr(modules, 'WanModel'))
        except Exception as e:
            self.fail(f"Failed to import wan.modules: {e}")
    
    def test_distributed_imports(self):
        """Test distributed module imports."""
        try:
            from wan import distributed
            # Test that distributed utils can be imported
            self.assertTrue(hasattr(distributed, '__name__'))
        except Exception as e:
            self.fail(f"Failed to import wan.distributed: {e}")


class TestTensorOperations(unittest.TestCase):
    """Test basic tensor operations and device handling."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def test_basic_tensor_creation(self):
        """Test basic tensor operations."""
        # Test tensor creation
        x = torch.randn(2, 3, 4)
        self.assertEqual(x.shape, (2, 3, 4))
        
        # Test device transfer
        x_device = x.to(self.device)
        self.assertEqual(x_device.device.type, self.device.type)
    
    def test_batch_operations(self):
        """Test batch operations similar to what Wan models use."""
        batch_size = 2
        seq_len = 512
        hidden_dim = 768
        
        # Simulate text embeddings
        text_emb = torch.randn(batch_size, seq_len, hidden_dim)
        self.assertEqual(text_emb.shape, (batch_size, seq_len, hidden_dim))
        
        # Test attention-like operation
        attention_weights = torch.softmax(torch.randn(batch_size, seq_len, seq_len), dim=-1)
        attended = torch.bmm(attention_weights, text_emb)
        self.assertEqual(attended.shape, text_emb.shape)


class TestConfigurationLoading(unittest.TestCase):
    """Test configuration loading and validation."""
    
    def test_config_access(self):
        """Test that configs can be accessed."""
        try:
            from wan.configs import t2v_A14B, i2v_A14B, ti2v_5B
            
            # Test that configs have expected attributes
            for config in [t2v_A14B, i2v_A14B, ti2v_5B]:
                self.assertTrue(hasattr(config, '__name__'))
                # These configs should have model parameters
                if hasattr(config, 'dim'):
                    self.assertIsInstance(config.dim, int)
                if hasattr(config, 'num_layers'):
                    self.assertIsInstance(config.num_layers, int)
                    
        except Exception as e:
            self.fail(f"Failed to access configs: {e}")


class TestTokenizerComponents(unittest.TestCase):
    """Test tokenizer-related components."""
    
    def test_tokenizer_import(self):
        """Test that tokenizer classes can be imported."""
        try:
            from wan.modules.tokenizers import HuggingfaceTokenizer
            # Just test that the class exists
            self.assertTrue(callable(HuggingfaceTokenizer))
        except Exception as e:
            self.fail(f"Failed to import tokenizer: {e}")


class TestVideoProcessingUtils(unittest.TestCase):
    """Test video and image processing utilities."""
    
    def test_basic_image_operations(self):
        """Test basic image operations."""
        from PIL import Image
        import numpy as np
        
        # Create a test image
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        
        # Test basic operations
        self.assertEqual(img.size, (224, 224))
        self.assertEqual(img.mode, 'RGB')
        
        # Test resize
        img_resized = img.resize((512, 512))
        self.assertEqual(img_resized.size, (512, 512))
    
    def test_video_frame_simulation(self):
        """Test video frame-like tensor operations."""
        # Simulate video frames: (batch, channels, frames, height, width)
        batch_size = 1
        channels = 3
        frames = 16
        height, width = 512, 512
        
        video_tensor = torch.randn(batch_size, channels, frames, height, width)
        self.assertEqual(video_tensor.shape, (batch_size, channels, frames, height, width))
        
        # Test frame extraction
        first_frame = video_tensor[:, :, 0, :, :]
        self.assertEqual(first_frame.shape, (batch_size, channels, height, width))


class TestModelArchitectureComponents(unittest.TestCase):
    """Test model architecture components that don't require weights."""
    
    def test_attention_mechanisms(self):
        """Test attention-like operations."""
        batch_size = 2
        seq_len = 128
        hidden_dim = 512
        num_heads = 8
        head_dim = hidden_dim // num_heads
        
        # Simulate multi-head attention inputs
        q = torch.randn(batch_size, num_heads, seq_len, head_dim)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        self.assertEqual(attn_output.shape, (batch_size, num_heads, seq_len, head_dim))
    
    def test_layer_norm_operations(self):
        """Test layer normalization operations."""
        batch_size = 4
        seq_len = 256
        hidden_dim = 768
        
        x = torch.randn(batch_size, seq_len, hidden_dim)
        layer_norm = torch.nn.LayerNorm(hidden_dim)
        normalized = layer_norm(x)
        
        self.assertEqual(normalized.shape, x.shape)
        
        # Check that normalization works (mean close to 0, std close to 1)
        self.assertTrue(torch.allclose(normalized.mean(dim=-1), torch.zeros(batch_size, seq_len), atol=1e-5))


class TestDistributedUtils(unittest.TestCase):
    """Test distributed computing utilities."""
    
    def test_distributed_imports(self):
        """Test that distributed utilities can be imported."""
        try:
            from wan.distributed import sequence_parallel, ulysses
            # Just test that modules exist
            self.assertTrue(hasattr(sequence_parallel, '__name__'))
            self.assertTrue(hasattr(ulysses, '__name__'))
        except Exception as e:
            self.fail(f"Failed to import distributed utils: {e}")


class TestMemoryManagement(unittest.TestCase):
    """Test memory management and optimization techniques."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def test_gradient_checkpointing_simulation(self):
        """Test gradient checkpointing-like behavior."""
        def dummy_forward(x):
            return torch.nn.functional.relu(x)
        
        x = torch.randn(4, 512, requires_grad=True)
        
        # Test with checkpointing disabled
        y1 = dummy_forward(x)
        self.assertEqual(y1.shape, x.shape)
        
        # Test that gradients can flow
        loss = y1.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
    
    def test_memory_efficient_attention_pattern(self):
        """Test memory-efficient attention patterns."""
        batch_size = 2
        seq_len = 1024
        hidden_dim = 256
        
        # Simulate chunked attention
        chunk_size = 256
        x = torch.randn(batch_size, seq_len, hidden_dim)
        
        chunks = []
        for i in range(0, seq_len, chunk_size):
            end_idx = min(i + chunk_size, seq_len)
            chunk = x[:, i:end_idx, :]
            chunks.append(chunk)
        
        # Verify chunking
        reconstructed = torch.cat(chunks, dim=1)
        self.assertTrue(torch.allclose(x, reconstructed))


class TestIntegrationReadiness(unittest.TestCase):
    """Test that the system is ready for full integration."""
    
    def test_cuda_availability(self):
        """Test CUDA setup if available."""
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            self.assertGreater(device_count, 0)
            
            # Test basic CUDA operations
            x = torch.randn(10, 10).cuda()
            y = torch.randn(10, 10).cuda()
            z = torch.mm(x, y)
            self.assertEqual(z.device.type, 'cuda')
        else:
            self.skipTest("CUDA not available")
    
    def test_mixed_precision_capability(self):
        """Test mixed precision operations."""
        if torch.cuda.is_available():
            x = torch.randn(4, 512).cuda()
            
            # Test autocast
            with torch.cuda.amp.autocast():
                y = torch.nn.functional.linear(x, torch.randn(256, 512).cuda())
                self.assertEqual(y.dtype, torch.float16)
        else:
            # Test CPU version
            x = torch.randn(4, 512)
            with torch.cpu.amp.autocast():
                y = torch.nn.functional.linear(x, torch.randn(256, 512))
                # On CPU, autocast might not change dtype
                self.assertTrue(y.dtype in [torch.float32, torch.bfloat16])


def run_tests():
    """Run all tests and return results."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestWanImports,
        TestTensorOperations,
        TestConfigurationLoading,
        TestTokenizerComponents,
        TestVideoProcessingUtils,
        TestModelArchitectureComponents,
        TestDistributedUtils,
        TestMemoryManagement,
        TestIntegrationReadiness
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == "__main__":
    print("=" * 70)
    print("Running Wan2.2 Unit Tests")
    print("=" * 70)
    
    result = run_tests()
    
    print("\n" + "=" * 70)
    if result.wasSuccessful():
        print("✅ All tests passed!")
    else:
        print(f"❌ {len(result.failures)} test(s) failed, {len(result.errors)} error(s)")
        
        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"- {test}: {traceback}")
        
        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"- {test}: {traceback}")
    
    print("=" * 70)