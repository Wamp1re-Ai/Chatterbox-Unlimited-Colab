"""
Fast setup script using UV package manager
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
    """Main setup function with UV"""
    print("⚡ ChatterBox Unlimited Setup with UV")
    print("=" * 45)
    
    # Check if we're in Colab
    try:
        import google.colab
        IN_COLAB = True
        print("☁️ Running in Google Colab")
    except ImportError:
        IN_COLAB = False
        print("💻 Running locally")
    
    # Install UV
    print("\n📦 Installing UV package manager...")
    run_command("pip install uv", "Installing UV")
    
    # Fix CUDA conflicts in Colab
    if IN_COLAB:
        print("\n🔧 Fixing CUDA version conflicts...")
        run_command("uv pip uninstall -y torch torchvision torchaudio", "Uninstalling existing PyTorch components")
        run_command("uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121", "Installing compatible PyTorch, Torchvision, Torchaudio")
        run_command("uv pip install \"numpy>=1.24.0,<2.0.0\"", "Installing compatible NumPy")
    else:
        print("\n🔥 Installing PyTorch with UV...")
        # Check for CUDA locally
        try:
            import torch
            if torch.cuda.is_available():
                run_command("uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118", "Installing PyTorch CUDA")
            else:
                run_command("uv pip install torch torchaudio", "Installing PyTorch CPU")
        except ImportError:
            run_command("uv pip install torch torchaudio", "Installing PyTorch")
    
    # Install ChatterBox TTS
    print("\n🎤 Installing ChatterBox TTS with UV...")
    if run_command("uv pip install chatterbox-tts --no-deps", "Installing ChatterBox TTS"):
        print("✅ ChatterBox TTS installed")
    else:
        print("🔄 Trying GitHub installation...")
        run_command("uv pip install git+https://github.com/resemble-ai/chatterbox.git", "Installing from GitHub")
    
    # Install remaining dependencies
    print("\n📚 Installing dependencies with UV...")
    deps = [
        "gradio",
        "soundfile", 
        "librosa",
        "resampy",
        "transformers",
        "diffusers",
        "omegaconf",
        "conformer",
        "resemble-perth",
        "s3tokenizer"
    ]
    # Ensure torchvision is installed if not handled by the specific Colab command
    if not IN_COLAB:
        deps.append("torchvision")
    
    for dep in deps:
        run_command(f"uv pip install {dep}", f"Installing {dep}")
    
    # Test installation
    print("\n🧪 Testing installation...")
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA Version: {torch.version.cuda}")
    except Exception as e:
        print(f"❌ PyTorch test failed: {e}")
        return False
    
    try:
        from chatterbox.tts import ChatterboxTTS
        print("✅ ChatterBox TTS ready")
    except Exception as e:
        print(f"❌ ChatterBox TTS test failed: {e}")
        print("🔄 You may need to restart your runtime/kernel")
        return False
    
    print("\n🎉 Setup completed successfully with UV!")
    print("\n🚀 Run: python main.py")
    if IN_COLAB:
        print("   Or: python main.py --share")
    
    return True

if __name__ == "__main__":
    main()
