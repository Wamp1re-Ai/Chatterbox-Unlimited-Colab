"""
Quick fix for common Colab issues with ChatterBox TTS
Addresses NumPy binary incompatibility and indentation errors
"""
import subprocess
import sys
import os

def fix_numpy_issue():
    """Fix NumPy binary incompatibility"""
    print("🔧 Fixing NumPy binary incompatibility...")
    
    # Uninstall and reinstall NumPy with specific version
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "numpy", "-y"], check=False)
    subprocess.run([sys.executable, "-m", "pip", "install", "numpy==1.24.4"], check=False)
    
    print("✅ NumPy fixed. You may need to restart runtime.")

def fix_transformers_issue():
    """Fix transformers compatibility"""
    print("🔧 Fixing transformers compatibility...")
    
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "transformers>=4.45.0"], check=False)
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "diffusers>=0.30.0"], check=False)
    
    print("✅ Transformers updated.")

def test_imports():
    """Test if imports work"""
    print("🧪 Testing imports...")
    
    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__} working")
    except Exception as e:
        print(f"❌ NumPy issue: {e}")
        return False
    
    try:
        from transformers.utils import is_quanto_available
        print("✅ Transformers compatibility OK")
    except Exception as e:
        print(f"❌ Transformers issue: {e}")
        return False
    
    try:
        from chatterbox.tts import ChatterboxTTS
        print("✅ ChatterBox TTS import successful")
        return True
    except Exception as e:
        print(f"❌ ChatterBox TTS issue: {e}")
        return False

def main():
    """Main fix function"""
    print("🚀 Quick Colab Fix for ChatterBox TTS")
    print("=" * 40)
    
    # Test current state
    if test_imports():
        print("🎉 Everything is working! No fixes needed.")
        return
    
    print("\n🔧 Applying fixes...")
    
    # Fix NumPy
    fix_numpy_issue()
    
    # Fix transformers
    fix_transformers_issue()
    
    print("\n🧪 Testing after fixes...")
    if test_imports():
        print("🎉 All fixes successful!")
    else:
        print("⚠️ Some issues remain. Try restarting runtime.")

if __name__ == "__main__":
    main()
