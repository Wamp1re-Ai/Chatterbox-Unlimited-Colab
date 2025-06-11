#!/usr/bin/env python3
"""
Test script for the fixed voice cloning implementation
"""

import os
import sys
import torch
from chatterbox.tts import ChatterboxTTS
from ChatterBox_TTS_Fixed_Voice_Cloning import generate_speech_fixed

def main():
    print("🧪 Testing ChatterBox TTS Voice Cloning Fixes")
    print("=" * 50)
    
    # Check CUDA availability
    if torch.cuda.is_available():
        device = "cuda"
        print(f"🎮 Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("💻 Using CPU")
    
    # Load model
    print("\n🤖 Loading ChatterBox TTS model...")
    try:
        model = ChatterboxTTS.from_pretrained(device=device)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Test 1: Standard TTS (should use parallel processing)
    print("\n" + "="*50)
    print("🧪 TEST 1: Standard TTS (Parallel Processing)")
    print("="*50)
    
    test_text = "Hello! This is a test of the standard text-to-speech functionality. It should process quickly using parallel processing since no voice cloning is involved."
    
    try:
        output_path, status = generate_speech_fixed(
            model=model,
            text=test_text,
            audio_file=None,  # No voice cloning
            exaggeration=0.5,
            cfg_weight=0.5
        )
        
        if output_path:
            print(f"✅ Standard TTS test passed!")
            print(f"📁 Output: {output_path}")
            print(f"📊 Status: {status}")
        else:
            print(f"❌ Standard TTS test failed: {status}")
            
    except Exception as e:
        print(f"❌ Standard TTS test error: {e}")
    
    # Test 2: Voice Cloning (should use sequential processing)
    print("\n" + "="*50)
    print("🧪 TEST 2: Voice Cloning (Sequential Processing)")
    print("="*50)
    
    # Check for reference audio files
    reference_files = ["reference.wav", "sample.wav", "voice.wav", "test_audio.wav"]
    reference_audio = None
    
    for ref_file in reference_files:
        if os.path.exists(ref_file):
            reference_audio = ref_file
            break
    
    if reference_audio:
        print(f"🎭 Found reference audio: {reference_audio}")
        
        clone_text = "This is a test of voice cloning functionality. The system should use sequential processing to avoid infinite loops and CUDA conflicts."
        
        try:
            output_path, status = generate_speech_fixed(
                model=model,
                text=clone_text,
                audio_file=reference_audio,  # Voice cloning enabled
                exaggeration=0.5,
                cfg_weight=0.5
            )
            
            if output_path:
                print(f"✅ Voice cloning test passed!")
                print(f"📁 Output: {output_path}")
                print(f"📊 Status: {status}")
            else:
                print(f"❌ Voice cloning test failed: {status}")
                
        except Exception as e:
            print(f"❌ Voice cloning test error: {e}")
    else:
        print("⚠️ No reference audio found for voice cloning test")
        print("💡 To test voice cloning, place a WAV file named:")
        for ref_file in reference_files:
            print(f"   - {ref_file}")
    
    # Test 3: Long text (should chunk properly)
    print("\n" + "="*50)
    print("🧪 TEST 3: Long Text Processing")
    print("="*50)
    
    long_text = """
    This is a comprehensive test of the ChatterBox TTS system with a longer piece of text that should be automatically chunked into smaller segments for processing. The system should handle this text efficiently by breaking it down into manageable pieces, processing each chunk appropriately, and then concatenating the results into a seamless audio output. This test verifies that the chunking algorithm works correctly and that the audio quality remains high even with longer inputs. The text should be split at natural sentence boundaries to maintain the flow and coherence of the speech output.
    """
    
    try:
        output_path, status = generate_speech_fixed(
            model=model,
            text=long_text.strip(),
            audio_file=None,
            exaggeration=0.4,
            cfg_weight=0.6
        )
        
        if output_path:
            print(f"✅ Long text test passed!")
            print(f"📁 Output: {output_path}")
            print(f"📊 Status: {status}")
        else:
            print(f"❌ Long text test failed: {status}")
            
    except Exception as e:
        print(f"❌ Long text test error: {e}")
    
    print("\n" + "="*50)
    print("🎉 Testing Complete!")
    print("="*50)
    
    print("\n📋 Key Improvements:")
    print("✅ Sequential processing for voice cloning (prevents infinite loops)")
    print("✅ Timeout protection (60s per chunk, 5min total)")
    print("✅ Smart text chunking at sentence boundaries")
    print("✅ Improved error handling and recovery")
    print("✅ Automatic fallback strategies")
    print("✅ Better user feedback and progress indicators")
    
    print("\n💡 Usage Tips:")
    print("- Voice cloning now uses sequential processing for stability")
    print("- Standard TTS uses parallel processing for speed")
    print("- Timeouts prevent infinite loops")
    print("- Text is automatically chunked for optimal processing")

if __name__ == "__main__":
    main()
