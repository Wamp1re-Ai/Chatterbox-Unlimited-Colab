"""
Fix for Colab dependency issues with ChatterBox TTS
Addresses the 'is_quanto_available' import error and other compatibility issues
"""
import subprocess
import sys
import importlib

def run_command(command):
    """Run a shell command and return the result"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def check_and_fix_transformers():
    """Check and fix transformers compatibility issues"""
    print("🔧 Checking transformers compatibility...")
    
    try:
        import transformers
        print(f"📦 Current transformers version: {transformers.__version__}")
        
        # Check if is_quanto_available is available
        try:
            from transformers.utils import is_quanto_available
            print("✅ is_quanto_available is available")
            return True
        except ImportError:
            print("❌ is_quanto_available not found - upgrading transformers...")
            
            # Upgrade transformers
            success, stdout, stderr = run_command("pip install --upgrade 'transformers>=4.45.0'")
            if success:
                print("✅ Transformers upgraded successfully")
                
                # Reload transformers
                importlib.reload(transformers)
                
                # Check again
                try:
                    from transformers.utils import is_quanto_available
                    print("✅ is_quanto_available now available after upgrade")
                    return True
                except ImportError:
                    print("❌ Still not available after upgrade")
                    return False
            else:
                print(f"❌ Failed to upgrade transformers: {stderr}")
                return False
                
    except ImportError:
        print("❌ Transformers not installed")
        return False

def check_and_fix_torch():
    """Check and fix PyTorch compatibility"""
    print("🔧 Checking PyTorch compatibility...")
    
    try:
        import torch
        print(f"🔥 PyTorch version: {torch.__version__}")
        print(f"🎮 CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA Version: {torch.version.cuda}")
        
        return True
    except ImportError:
        print("❌ PyTorch not installed properly")
        return False

def install_chatterbox_tts():
    """Install ChatterBox TTS with proper error handling"""
    print("🎤 Installing ChatterBox TTS...")
    
    # Try standard installation first
    success, stdout, stderr = run_command("pip install chatterbox-tts")
    
    if success:
        print("✅ ChatterBox TTS installed via pip")
    else:
        print("❌ Pip installation failed, trying GitHub...")
        success, stdout, stderr = run_command("pip install git+https://github.com/resemble-ai/chatterbox.git")
        
        if success:
            print("✅ ChatterBox TTS installed from GitHub")
        else:
            print(f"❌ GitHub installation failed: {stderr}")
            return False
    
    # Test the installation
    try:
        from chatterbox.tts import ChatterboxTTS
        print("✅ ChatterBox TTS import successful!")
        return True
    except ImportError as e:
        print(f"❌ ChatterBox TTS import failed: {e}")
        return False

def install_enhanced_dependencies():
    """Install dependencies for enhanced features"""
    print("📦 Installing enhanced feature dependencies...")
    
    dependencies = [
        "numpy>=1.24.0,<2.0.0",
        "scipy>=1.10.0",
        "soundfile>=0.12.1",
        "librosa>=0.10.0",
        "gradio>=4.0.0"
    ]
    
    for dep in dependencies:
        print(f"   Installing {dep}...")
        success, stdout, stderr = run_command(f"pip install '{dep}'")
        if success:
            print(f"   ✅ {dep} installed")
        else:
            print(f"   ❌ Failed to install {dep}: {stderr}")

def main():
    """Main fix function"""
    print("🚀 ChatterBox TTS Colab Dependency Fix")
    print("=" * 50)
    
    # Step 1: Fix transformers
    print("\n1️⃣ Fixing transformers compatibility...")
    transformers_ok = check_and_fix_transformers()
    
    # Step 2: Check PyTorch
    print("\n2️⃣ Checking PyTorch...")
    torch_ok = check_and_fix_torch()
    
    # Step 3: Install enhanced dependencies
    print("\n3️⃣ Installing enhanced dependencies...")
    install_enhanced_dependencies()
    
    # Step 4: Install ChatterBox TTS
    print("\n4️⃣ Installing ChatterBox TTS...")
    chatterbox_ok = install_chatterbox_tts()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Installation Summary:")
    print(f"   Transformers: {'✅' if transformers_ok else '❌'}")
    print(f"   PyTorch: {'✅' if torch_ok else '❌'}")
    print(f"   ChatterBox TTS: {'✅' if chatterbox_ok else '❌'}")
    
    if all([transformers_ok, torch_ok, chatterbox_ok]):
        print("\n🎉 All dependencies fixed successfully!")
        print("You can now run the ChatterBox TTS interface.")
        
        # Test enhanced features
        print("\n🧪 Testing enhanced features...")
        try:
            # Test if our enhanced modules can be imported
            import os
            if os.path.exists('emotion_presets.py'):
                from emotion_presets import emotion_presets
                print("✅ Emotion presets available")
            
            if os.path.exists('audio_processor.py'):
                from audio_processor import audio_processor
                print("✅ Audio processor available")
            
            if os.path.exists('text_processor.py'):
                from text_processor import text_processor
                print("✅ Text processor available")
                
            print("🎊 Enhanced features are ready!")
            
        except Exception as e:
            print(f"⚠️ Enhanced features not available: {e}")
            print("   (This is normal if you haven't cloned the enhanced repository)")
        
    else:
        print("\n❌ Some issues remain. You may need to:")
        print("   1. Restart the Colab runtime")
        print("   2. Run this script again")
        print("   3. Check for any error messages above")

if __name__ == "__main__":
    main()
