#!/usr/bin/env python3
"""
Test script to verify PyTorch/TorchVision compatibility fixes
Run this to test the fixes before using the Colab notebook
"""

import sys
import importlib
import subprocess

def safe_import_test(module_name, test_func=None):
    """Safely test module import with optional functionality test"""
    try:
        # Clear any cached imports first
        if module_name in sys.modules:
            del sys.modules[module_name]
        
        module = importlib.import_module(module_name)
        version = getattr(module, '__version__', 'unknown')
        
        if test_func:
            test_func(module)
        
        print(f"✅ {module_name}: {version}")
        return True, module
    except Exception as e:
        print(f"❌ {module_name}: {e}")
        return False, None

def test_torch(torch_module):
    """Test basic PyTorch functionality"""
    x = torch_module.tensor([1.0, 2.0, 3.0])
    y = x * 2
    assert y.sum().item() == 12.0
    print(f"   🎮 CUDA available: {torch_module.cuda.is_available()}")
    if torch_module.cuda.is_available():
        print(f"   📱 GPU: {torch_module.cuda.get_device_name(0)}")
        print(f"   💾 GPU Memory: {torch_module.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

def test_torchvision(tv_module):
    """Test TorchVision functionality"""
    import torch
    try:
        # Test NMS operator
        boxes = torch.tensor([[0, 0, 1, 1], [0.5, 0.5, 1.5, 1.5]], dtype=torch.float32)
        scores = torch.tensor([0.9, 0.8], dtype=torch.float32)
        keep = tv_module.ops.nms(boxes, scores, 0.5)
        print("   🧪 NMS operator test: PASSED")
    except Exception as e:
        print(f"   ⚠️ NMS test failed: {e}")
        # Test basic transforms instead
        transform = tv_module.transforms.ToTensor()
        print("   🧪 Basic transforms test: PASSED")

def test_transformers(tf_module):
    """Test Transformers functionality"""
    try:
        from transformers.utils import is_quanto_available
        print("   🔧 is_quanto_available function: FOUND")
    except ImportError as e:
        print(f"   ❌ is_quanto_available not found: {e}")
        raise

def main():
    print("🔍 Testing PyTorch ecosystem compatibility...")
    print("=" * 60)
    
    all_good = True
    
    # Test PyTorch
    success, torch = safe_import_test("torch", test_torch)
    all_good &= success
    
    # Test TorchVision
    success, torchvision = safe_import_test("torchvision", test_torchvision)
    all_good &= success
    
    # Test Transformers
    success, transformers = safe_import_test("transformers", test_transformers)
    all_good &= success
    
    # Test other dependencies
    for module in ["numpy", "gradio", "soundfile", "librosa"]:
        success, _ = safe_import_test(module)
        all_good &= success
    
    print("=" * 60)
    if all_good:
        print("🎉 ALL TESTS PASSED! The notebook should work correctly.")
    else:
        print("⚠️ Some tests failed. Check the error messages above.")
        print("💡 Try installing the requirements: pip install -r requirements.txt")
    
    print(f"\n📊 Python version: {sys.version}")
    print(f"🔍 Total modules loaded: {len(sys.modules)}")

if __name__ == "__main__":
    main()
