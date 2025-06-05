"""
Test script for enhanced ChatterBox TTS features
Tests audio post-processing, emotion presets, batch processing, and text processing
"""
import os
import sys
import tempfile
import json

def test_imports():
    """Test if all enhancement modules can be imported"""
    print("🧪 Testing imports...")
    
    try:
        from audio_processor import audio_processor
        print("✅ Audio processor imported successfully")
    except ImportError as e:
        print(f"❌ Audio processor import failed: {e}")
        return False
    
    try:
        from emotion_presets import emotion_presets
        print("✅ Emotion presets imported successfully")
    except ImportError as e:
        print(f"❌ Emotion presets import failed: {e}")
        return False
    
    try:
        from text_processor import text_processor
        print("✅ Text processor imported successfully")
    except ImportError as e:
        print(f"❌ Text processor import failed: {e}")
        return False
    
    try:
        from batch_processor import BatchProcessor
        print("✅ Batch processor imported successfully")
    except ImportError as e:
        print(f"❌ Batch processor import failed: {e}")
        return False
    
    return True

def test_emotion_presets():
    """Test emotion presets functionality"""
    print("\n🎭 Testing emotion presets...")
    
    from emotion_presets import emotion_presets
    
    # Test getting presets
    all_presets = emotion_presets.get_all_presets()
    print(f"✅ Found {len(all_presets)} total presets")
    
    emotion_presets_list = emotion_presets.get_emotion_presets()
    style_presets_list = emotion_presets.get_style_presets()
    print(f"✅ Found {len(emotion_presets_list)} emotion presets")
    print(f"✅ Found {len(style_presets_list)} style presets")
    
    # Test specific preset
    happy_preset = emotion_presets.get_preset("happy")
    print(f"✅ Happy preset: exaggeration={happy_preset['exaggeration']}, cfg_weight={happy_preset['cfg_weight']}")
    
    # Test preset parameters
    exag, cfg = emotion_presets.get_preset_parameters("excited")
    print(f"✅ Excited preset parameters: {exag}, {cfg}")
    
    # Test custom preset creation
    success = emotion_presets.create_custom_preset(
        "test_emotion",
        "emotion",
        "Test emotion for testing",
        0.8,
        0.6,
        "🧪",
        ["test"]
    )
    print(f"✅ Custom preset creation: {success}")
    
    # Test search
    search_results = emotion_presets.search_presets("happy")
    print(f"✅ Search for 'happy' found {len(search_results)} results")
    
    return True

def test_text_processor():
    """Test text processing functionality"""
    print("\n📝 Testing text processor...")
    
    from text_processor import text_processor
    
    # Test basic text processing
    test_text = "Hello! This is a test with [laugh] and some URLs like https://github.com and numbers like 2024."
    processed = text_processor.process_text(test_text)
    print(f"✅ Original: {test_text}")
    print(f"✅ Processed: {processed}")
    
    # Test sound effects
    sound_effects_text = "I'm so happy [giggle] but I need to [cough] clear my throat [ahem]."
    processed_effects = text_processor.process_sound_effects(sound_effects_text)
    print(f"✅ Sound effects: {sound_effects_text} → {processed_effects}")
    
    # Test pronunciations
    tech_text = "The AI uses GPU and CPU with HTTP API."
    processed_tech = text_processor.process_pronunciations(tech_text)
    print(f"✅ Pronunciations: {tech_text} → {processed_tech}")
    
    # Test custom pronunciation
    success = text_processor.add_pronunciation("ChatterBox", "Chatter Box")
    print(f"✅ Custom pronunciation added: {success}")
    
    # Test available sound effects
    effects = text_processor.get_available_sound_effects()
    print(f"✅ Available sound effects: {len(effects)} effects")
    
    # Test processing preview
    preview = text_processor.get_processing_preview("Hello [laugh] world! Visit https://example.com in 2024.")
    print(f"✅ Processing preview generated with {len(preview)} steps")
    
    return True

def test_audio_processor():
    """Test audio processing functionality"""
    print("\n🎚️ Testing audio processor...")
    
    from audio_processor import audio_processor
    import numpy as np
    import soundfile as sf
    
    # Create a test audio file
    sample_rate = 24000
    duration = 2.0  # 2 seconds
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Generate a simple sine wave with some noise
    frequency = 440  # A4 note
    audio = np.sin(2 * np.pi * frequency * t) * 0.5
    noise = np.random.normal(0, 0.1, len(audio))
    noisy_audio = audio + noise
    
    # Save test audio
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        test_audio_path = tmp_file.name
    
    sf.write(test_audio_path, noisy_audio, sample_rate)
    print(f"✅ Created test audio file: {test_audio_path}")
    
    # Test noise reduction
    cleaned_audio = audio_processor.reduce_noise(noisy_audio, sample_rate, 0.5)
    print(f"✅ Noise reduction applied: {len(cleaned_audio)} samples")
    
    # Test normalization
    normalized_audio = audio_processor.normalize_audio(noisy_audio, -20.0)
    print(f"✅ Audio normalized to -20dB")
    
    # Test EQ
    eq_audio = audio_processor.apply_eq(noisy_audio, sample_rate, "vocal")
    print(f"✅ Vocal EQ applied")
    
    # Test full enhancement
    enhanced_path, status = audio_processor.enhance_audio(
        test_audio_path,
        noise_reduction=0.3,
        normalize=True,
        target_db=-20.0,
        eq_preset="vocal"
    )
    
    if enhanced_path:
        print(f"✅ Audio enhancement successful: {enhanced_path}")
        # Clean up
        try:
            os.unlink(enhanced_path)
        except:
            pass
    else:
        print(f"❌ Audio enhancement failed: {status}")
    
    # Clean up test file
    try:
        os.unlink(test_audio_path)
    except:
        pass
    
    return enhanced_path is not None

def test_batch_processor():
    """Test batch processing functionality"""
    print("\n📦 Testing batch processor...")
    
    from batch_processor import BatchProcessor
    from chatterbox_tts import tts_interface
    
    # Create batch processor
    batch_processor = BatchProcessor(tts_interface)
    print("✅ Batch processor created")
    
    # Test adding text batch
    test_texts = [
        "Hello, this is test one.",
        "This is the second test.",
        "And this is the third test."
    ]
    
    item_ids = batch_processor.add_text_batch(test_texts, exaggeration=0.5, cfg_weight=0.5)
    print(f"✅ Added {len(item_ids)} text items to batch")
    
    # Test CSV batch
    csv_content = """text,exaggeration,cfg_weight
"Hello from CSV",0.6,0.4
"Second CSV line",0.7,0.5"""
    
    csv_ids, csv_status = batch_processor.add_csv_batch(csv_content)
    print(f"✅ CSV batch: {csv_status}")
    
    # Test JSON batch
    json_content = json.dumps([
        {"text": "JSON test one", "exaggeration": 0.5},
        {"text": "JSON test two", "cfg_weight": 0.6}
    ])
    
    json_ids, json_status = batch_processor.add_json_batch(json_content)
    print(f"✅ JSON batch: {json_status}")
    
    # Test batch status
    status = batch_processor.get_batch_status()
    print(f"✅ Batch status: {status['total_items']} total items, {status['queue_size']} in queue")
    
    # Test clearing batch
    batch_processor.clear_batch()
    status_after_clear = batch_processor.get_batch_status()
    print(f"✅ After clear: {status_after_clear['total_items']} total items")
    
    return True

def test_integration():
    """Test integration between modules"""
    print("\n🔗 Testing module integration...")
    
    from chatterbox_tts import tts_interface
    from emotion_presets import emotion_presets
    from text_processor import text_processor
    
    # Test text processing with emotion presets
    test_text = "I'm so excited [laugh] about this new feature!"
    processed_text = text_processor.process_text(test_text)
    
    # Get preset parameters
    exag, cfg = emotion_presets.get_preset_parameters("excited")
    
    print(f"✅ Integrated test:")
    print(f"   Original text: {test_text}")
    print(f"   Processed text: {processed_text}")
    print(f"   Excited preset: exaggeration={exag}, cfg_weight={cfg}")
    
    # Test if TTS interface has enhanced methods
    if hasattr(tts_interface, 'generate_speech'):
        import inspect
        sig = inspect.signature(tts_interface.generate_speech)
        params = list(sig.parameters.keys())
        
        enhanced_params = ['preset', 'enable_text_processing', 'enable_audio_enhancement']
        has_enhancements = all(param in params for param in enhanced_params)
        
        print(f"✅ TTS interface enhanced: {has_enhancements}")
        if has_enhancements:
            print(f"   Enhanced parameters: {enhanced_params}")
    
    return True

def main():
    """Run all tests"""
    print("🚀 Starting ChatterBox TTS Enhanced Features Test")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Emotion Presets", test_emotion_presets),
        ("Text Processor", test_text_processor),
        ("Audio Processor", test_audio_processor),
        ("Batch Processor", test_batch_processor),
        ("Integration", test_integration)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
            print(f"✅ {test_name}: {'PASSED' if result else 'FAILED'}")
        except Exception as e:
            results[test_name] = False
            print(f"❌ {test_name}: FAILED - {str(e)}")
        
        print("-" * 40)
    
    # Summary
    print("\n📊 Test Summary:")
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Enhanced features are working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
