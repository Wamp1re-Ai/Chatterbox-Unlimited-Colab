#!/usr/bin/env python3
"""
Simple ChatterBox TTS Demo
A minimal example showing how to use ChatterBox TTS with proper error handling
"""

import os
import sys
import torch
import torchaudio
import tempfile
from pathlib import Path

def check_environment():
    """Check if the environment is properly set up"""
    print("🔍 Checking environment...")
    
    # Check Python version
    print(f"Python: {sys.version}")
    
    # Check PyTorch
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"🎮 CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"📱 GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("❌ PyTorch not found")
        return False
    
    # Check ChatterBox TTS
    try:
        from chatterbox.tts import ChatterboxTTS
        print("✅ ChatterBox TTS: Available")
        return True
    except ImportError as e:
        print(f"❌ ChatterBox TTS: {e}")
        print("💡 Install with: pip install chatterbox-tts")
        return False

def load_model():
    """Load the ChatterBox TTS model"""
    print("\n🔄 Loading ChatterBox TTS model...")
    
    try:
        from chatterbox.tts import ChatterboxTTS
        
        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🎮 Using device: {device}")
        
        # Load model
        model = ChatterboxTTS.from_pretrained(device=device)
        print("✅ Model loaded successfully!")
        
        return model
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None

def generate_speech(model, text, output_path="output.wav", exaggeration=0.5, cfg_weight=0.5):
    """Generate speech from text"""
    print(f"\n🎤 Generating speech: '{text[:50]}...'")
    
    try:
        # Generate audio
        wav = model.generate(
            text,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight
        )
        
        # Save audio
        torchaudio.save(output_path, wav, model.sr)
        
        duration = wav.shape[1] / model.sr
        print(f"✅ Generated {duration:.1f}s of audio")
        print(f"💾 Saved to: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        return False

def clone_voice(model, text, reference_audio, output_path="cloned_output.wav", 
                exaggeration=0.5, cfg_weight=0.5):
    """Generate speech with voice cloning"""
    print(f"\n🎭 Cloning voice from: {reference_audio}")
    print(f"📝 Text: '{text[:50]}...'")
    
    try:
        # Generate audio with voice cloning
        wav = model.generate(
            text,
            audio_prompt_path=reference_audio,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight
        )
        
        # Save audio
        torchaudio.save(output_path, wav, model.sr)
        
        duration = wav.shape[1] / model.sr
        print(f"✅ Generated {duration:.1f}s of cloned audio")
        print(f"💾 Saved to: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Voice cloning failed: {e}")
        return False

def main():
    """Main demo function"""
    print("🎤 ChatterBox TTS - Simple Demo")
    print("=" * 40)
    
    # Check environment
    if not check_environment():
        print("\n❌ Environment check failed. Please install dependencies.")
        return
    
    # Load model
    model = load_model()
    if model is None:
        print("\n❌ Failed to load model. Exiting.")
        return
    
    # Demo texts
    demo_texts = [
        "Hello! This is ChatterBox TTS, a state-of-the-art text-to-speech model.",
        "I can generate natural-sounding speech from any text you provide.",
        "Try adjusting the exaggeration and CFG weight parameters for different effects!"
    ]
    
    print("\n🎯 Running TTS demos...")
    
    # Generate standard TTS samples
    for i, text in enumerate(demo_texts):
        output_path = f"demo_output_{i+1}.wav"
        
        # Vary parameters for different effects
        if i == 0:
            # Natural speech
            exaggeration, cfg_weight = 0.5, 0.5
        elif i == 1:
            # More expressive
            exaggeration, cfg_weight = 0.7, 0.3
        else:
            # Calm and deliberate
            exaggeration, cfg_weight = 0.3, 0.6
        
        print(f"\n--- Demo {i+1} (exaggeration={exaggeration}, cfg_weight={cfg_weight}) ---")
        generate_speech(model, text, output_path, exaggeration, cfg_weight)
    
    # Voice cloning demo (if reference audio exists)
    reference_files = ["reference.wav", "sample.wav", "voice.wav"]
    reference_audio = None
    
    for ref_file in reference_files:
        if os.path.exists(ref_file):
            reference_audio = ref_file
            break
    
    if reference_audio:
        print(f"\n🎭 Voice cloning demo with {reference_audio}")
        clone_text = "This is a demonstration of voice cloning using ChatterBox TTS."
        clone_voice(model, clone_text, reference_audio, "voice_clone_demo.wav")
    else:
        print(f"\n💡 To test voice cloning, place a reference audio file named:")
        print(f"   'reference.wav', 'sample.wav', or 'voice.wav' in this directory")
    
    print("\n🎉 Demo complete!")
    print("\n📁 Generated files:")
    for i in range(len(demo_texts)):
        output_file = f"demo_output_{i+1}.wav"
        if os.path.exists(output_file):
            print(f"   ✅ {output_file}")
    
    if reference_audio and os.path.exists("voice_clone_demo.wav"):
        print(f"   ✅ voice_clone_demo.wav")
    
    print("\n💡 Tips:")
    print("   - Use headphones for best audio quality")
    print("   - Generated audio includes watermarking for responsible AI use")
    print("   - Adjust exaggeration (0.0-1.0) for emotion intensity")
    print("   - Adjust cfg_weight (0.0-1.0) for speech pacing")

if __name__ == "__main__":
    main()
