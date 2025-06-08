#!/usr/bin/env python3
"""
CUDA Error Fix for ChatterBox TTS
Quick script to resolve "CUDA error: device-side assert triggered" issues
"""

import os
import sys
import torch
import tempfile
import librosa
import soundfile as sf
import numpy as np

def clear_cuda_cache():
    """Clear CUDA cache and synchronize"""
    if torch.cuda.is_available():
        print("🧹 Clearing CUDA cache...")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("✅ CUDA cache cleared")
    else:
        print("⚠️ CUDA not available")

def preprocess_audio_safe(audio_path, output_path=None):
    """
    Safely preprocess audio to prevent CUDA errors
    
    Args:
        audio_path: Path to input audio file
        output_path: Path for output (optional, creates temp file if None)
    
    Returns:
        str: Path to preprocessed audio file
    """
    try:
        print(f"🔍 Processing audio: {audio_path}")
        
        # Load audio with librosa
        audio, sr = librosa.load(audio_path, sr=None)
        print(f"📊 Original: {len(audio)/sr:.1f}s, {sr}Hz")
        
        # Ensure mono
        if audio.ndim > 1:
            audio = librosa.to_mono(audio)
        
        # Trim silence
        audio, _ = librosa.effects.trim(audio, top_db=20)
        
        # Normalize
        audio = librosa.util.normalize(audio)
        
        # Limit duration to prevent memory issues
        max_duration = 30  # seconds
        if len(audio) / sr > max_duration:
            audio = audio[:int(max_duration * sr)]
            print(f"⚠️ Trimmed to {max_duration}s")
        
        # Resample to ChatterBox expected rate
        target_sr = 22050
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
            print(f"🔄 Resampled to {sr}Hz")
        
        # Ensure proper amplitude range
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.8  # 80% of max to prevent clipping
        
        # Save processed audio
        if output_path is None:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                output_path = tmp.name
        
        sf.write(output_path, audio, sr)
        
        duration = len(audio) / sr
        print(f"✅ Processed: {duration:.1f}s, {sr}Hz → {output_path}")
        
        return output_path
        
    except Exception as e:
        print(f"❌ Audio processing failed: {e}")
        return None

def safe_generate_speech(model, text, audio_prompt_path=None, **kwargs):
    """
    Generate speech with CUDA error prevention
    
    Args:
        model: ChatterBox TTS model
        text: Text to synthesize
        audio_prompt_path: Optional audio for voice cloning
        **kwargs: Additional generation parameters
    
    Returns:
        torch.Tensor: Generated audio waveform
    """
    # Set CUDA debugging
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    try:
        # Clear cache before generation
        clear_cuda_cache()
        
        # Limit text length
        if len(text) > 500:
            text = text[:500]
            print(f"⚠️ Text truncated to 500 characters")
        
        # Set safe default parameters
        safe_params = {
            'exaggeration': kwargs.get('exaggeration', 0.5),
            'cfg_weight': kwargs.get('cfg_weight', 0.5)
        }
        
        # Attempt generation with retries
        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(f"🎤 Generation attempt {attempt + 1}/{max_retries}")
                
                if audio_prompt_path:
                    wav = model.generate(text, audio_prompt_path=audio_prompt_path, **safe_params)
                else:
                    wav = model.generate(text, **safe_params)
                
                print("✅ Generation successful!")
                return wav
                
            except RuntimeError as e:
                if "CUDA" in str(e) or "assert" in str(e):
                    print(f"⚠️ CUDA error on attempt {attempt + 1}: {str(e)[:100]}...")
                    
                    # Clear cache and adjust parameters
                    clear_cuda_cache()
                    
                    # Reduce text length
                    if len(text) > 50:
                        text = text[:len(text)//2]
                        print(f"🔄 Reduced text to {len(text)} characters")
                    
                    # Make parameters more conservative
                    safe_params['exaggeration'] = min(safe_params['exaggeration'], 0.3)
                    safe_params['cfg_weight'] = min(safe_params['cfg_weight'], 0.5)
                    print(f"🔧 Adjusted parameters: {safe_params}")
                    
                    if attempt < max_retries - 1:
                        continue
                
                raise e
        
    finally:
        # Clean up environment
        if 'CUDA_LAUNCH_BLOCKING' in os.environ:
            del os.environ['CUDA_LAUNCH_BLOCKING']

def diagnose_cuda_environment():
    """Diagnose CUDA environment for issues"""
    print("🔍 CUDA Environment Diagnosis")
    print("=" * 40)
    
    # Check PyTorch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name}")
            print(f"  Memory: {props.total_memory / 1e9:.1f} GB")
            print(f"  Compute capability: {props.major}.{props.minor}")
        
        # Check memory usage
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1e9
            memory_reserved = torch.cuda.memory_reserved() / 1e9
            print(f"Memory allocated: {memory_allocated:.1f} GB")
            print(f"Memory reserved: {memory_reserved:.1f} GB")
    
    print("\n🔧 Recommendations:")
    if not torch.cuda.is_available():
        print("❌ Enable GPU: Runtime → Change runtime type → GPU")
    else:
        print("✅ CUDA is available")
        
        memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        if memory_total < 4:
            print("⚠️ Low GPU memory - use shorter text and conservative parameters")
        else:
            print("✅ Sufficient GPU memory")

def main():
    """Main function for testing CUDA fixes"""
    print("🎤 ChatterBox TTS - CUDA Error Fix Utility")
    print("=" * 50)
    
    # Diagnose environment
    diagnose_cuda_environment()
    
    # Test CUDA operations
    print("\n🧪 Testing CUDA operations...")
    try:
        if torch.cuda.is_available():
            # Test basic CUDA operations
            x = torch.randn(100, 100).cuda()
            y = torch.matmul(x, x)
            print("✅ Basic CUDA operations work")
            
            # Clear cache
            clear_cuda_cache()
            print("✅ CUDA cache management works")
        else:
            print("⚠️ CUDA not available - tests skipped")
    except Exception as e:
        print(f"❌ CUDA test failed: {e}")
    
    print("\n💡 Usage Tips:")
    print("1. Use preprocess_audio_safe() for audio files")
    print("2. Use safe_generate_speech() instead of model.generate()")
    print("3. Call clear_cuda_cache() before generation")
    print("4. Keep text under 500 characters")
    print("5. Use conservative parameters (exaggeration < 0.5)")

if __name__ == "__main__":
    main()
