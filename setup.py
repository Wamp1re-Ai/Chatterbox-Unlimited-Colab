"""
Setup script for ChatterBox Unlimited
"""
import subprocess
import sys
import os

def run_command(command, description=""):
    """Run a command and return success status"""
    print(f"🔄 {description}")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} - Success")
            return True
        else:
            print(f"❌ {description} - Failed")
            if result.stderr.strip():
                print(f"   Error: {result.stderr.strip()}")
            return False
    except Exception as e:
        print(f"❌ {description} - Exception: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 ChatterBox Unlimited Setup")
    print("=" * 40)
    
    # Check if we're in Colab
    try:
        import google.colab
        IN_COLAB = True
        print("☁️ Running in Google Colab")
    except ImportError:
        IN_COLAB = False
        print("💻 Running locally")
    
    # Install PyTorch with CUDA
    if IN_COLAB:
        print("\n🔥 Installing PyTorch with CUDA for Colab...")
        run_command("pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118", "Installing PyTorch CUDA")
    else:
        print("\n🔥 Installing PyTorch...")
        run_command("pip install torch torchaudio", "Installing PyTorch")
    
    # Install ChatterBox TTS
    print("\n🎤 Installing ChatterBox TTS...")
    run_command("pip install chatterbox-tts", "Installing ChatterBox TTS")
    
    # Install other dependencies
    print("\n📦 Installing other dependencies...")
    deps = ["gradio", "soundfile", "librosa", "numpy"]
    for dep in deps:
        run_command(f"pip install {dep}", f"Installing {dep}")
    
    # Test installation
    print("\n🧪 Testing installation...")
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"❌ PyTorch test failed: {e}")
        return False
    
    try:
        from chatterbox.tts import ChatterboxTTS
        print("✅ ChatterBox TTS ready")
    except Exception as e:
        print(f"❌ ChatterBox TTS test failed: {e}")
        return False
    
    print("\n🎉 Setup completed successfully!")
    print("\n🚀 Run: python main.py")
    if IN_COLAB:
        print("   Or: python main.py --share")
    
    return True

if __name__ == "__main__":
    main()
