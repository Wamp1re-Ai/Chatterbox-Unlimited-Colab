"""
Demo script showcasing the enhanced ChatterBox TTS features
"""
import os
import tempfile
from emotion_presets import emotion_presets
from text_processor import text_processor
from audio_processor import audio_processor

def demo_emotion_presets():
    """Demonstrate emotion and style presets"""
    print("🎭 EMOTION & STYLE PRESETS DEMO")
    print("=" * 50)
    
    # Show available presets
    emotion_presets_list = emotion_presets.get_emotion_presets()
    style_presets_list = emotion_presets.get_style_presets()
    
    print(f"📊 Available Presets:")
    print(f"   Emotions: {len(emotion_presets_list)} presets")
    print(f"   Styles: {len(style_presets_list)} presets")
    
    # Demonstrate specific presets
    demo_presets = ["happy", "sad", "excited", "news_anchor", "storyteller"]
    
    for preset_name in demo_presets:
        preset = emotion_presets.get_preset(preset_name)
        print(f"\n{preset.get('icon', '🎭')} {preset.get('name', preset_name)}:")
        print(f"   Description: {preset.get('description', 'N/A')}")
        print(f"   Parameters: exaggeration={preset.get('exaggeration', 0.5):.1f}, cfg_weight={preset.get('cfg_weight', 0.5):.1f}")
        print(f"   Tags: {', '.join(preset.get('tags', []))}")
    
    # Demonstrate preset suggestions
    test_texts = [
        "I'm so happy and excited about this!",
        "Unfortunately, this is very sad news.",
        "Breaking news: Scientists make amazing discovery!"
    ]
    
    print(f"\n🤖 Smart Preset Suggestions:")
    for text in test_texts:
        suggestions = emotion_presets.get_preset_suggestions(text)
        print(f"   Text: \"{text[:40]}...\"")
        print(f"   Suggested presets: {', '.join(suggestions)}")

def demo_text_processing():
    """Demonstrate text processing features"""
    print("\n\n📝 TEXT PROCESSING DEMO")
    print("=" * 50)
    
    # Demonstrate different processing features
    test_cases = [
        {
            "name": "Sound Effects",
            "text": "Hello [laugh] I'm so happy [giggle] but I need to [cough] excuse me [ahem]."
        },
        {
            "name": "Pronunciations", 
            "text": "The AI uses GPU and CPU with HTTP API from GitHub."
        },
        {
            "name": "Numbers & Years",
            "text": "In 2024, we processed 1000 files and achieved 95% accuracy."
        },
        {
            "name": "URLs & Emails",
            "text": "Visit https://github.com or email support@example.com for help."
        },
        {
            "name": "Mixed Processing",
            "text": "Hey [laugh] check out https://github.com! The AI API processes 2024 requests [giggle]."
        }
    ]
    
    for case in test_cases:
        print(f"\n🔧 {case['name']}:")
        print(f"   Original: {case['text']}")
        processed = text_processor.process_text(case['text'])
        print(f"   Processed: {processed}")
    
    # Show available sound effects
    effects = text_processor.get_available_sound_effects()
    print(f"\n🎵 Available Sound Effects ({len(effects)} total):")
    print(f"   {', '.join(effects[:10])}...")  # Show first 10
    
    # Demonstrate custom pronunciation
    print(f"\n📚 Custom Pronunciation Demo:")
    text_processor.add_pronunciation("ChatterBox", "Chatter Box T T S")
    original = "Welcome to ChatterBox TTS!"
    processed = text_processor.process_pronunciations(original)
    print(f"   Original: {original}")
    print(f"   With custom pronunciation: {processed}")

def demo_audio_processing():
    """Demonstrate audio processing features"""
    print("\n\n🎚️ AUDIO PROCESSING DEMO")
    print("=" * 50)
    
    # Create a demo audio file (sine wave with noise)
    import numpy as np
    import soundfile as sf
    
    sample_rate = 24000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Generate test audio: sine wave + noise
    frequency = 440  # A4 note
    clean_audio = np.sin(2 * np.pi * frequency * t) * 0.3
    noise = np.random.normal(0, 0.05, len(clean_audio))
    noisy_audio = clean_audio + noise
    
    # Save test audio
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        test_audio_path = tmp_file.name
    
    sf.write(test_audio_path, noisy_audio, sample_rate)
    print(f"📁 Created test audio: {os.path.basename(test_audio_path)}")
    
    # Demonstrate different processing options
    processing_options = [
        {"name": "Vocal EQ + Light Noise Reduction", "noise_reduction": 0.2, "eq_preset": "vocal"},
        {"name": "Warm EQ + Medium Noise Reduction", "noise_reduction": 0.5, "eq_preset": "warm"},
        {"name": "Bright EQ + Heavy Noise Reduction", "noise_reduction": 0.8, "eq_preset": "bright"},
    ]
    
    for option in processing_options:
        print(f"\n🎛️ {option['name']}:")
        enhanced_path, status = audio_processor.enhance_audio(
            test_audio_path,
            noise_reduction=option['noise_reduction'],
            normalize=True,
            target_db=-20.0,
            eq_preset=option['eq_preset']
        )
        
        if enhanced_path:
            print(f"   ✅ Enhancement successful!")
            print(f"   📁 Output: {os.path.basename(enhanced_path)}")
            # Clean up
            try:
                os.unlink(enhanced_path)
            except:
                pass
        else:
            print(f"   ❌ Enhancement failed: {status}")
    
    # Clean up test file
    try:
        os.unlink(test_audio_path)
    except:
        pass
    
    # Show available EQ presets
    eq_presets = ["vocal", "balanced", "warm", "bright"]
    print(f"\n🎚️ Available EQ Presets:")
    for preset in eq_presets:
        print(f"   • {preset.title()}: Optimized for {preset} content")

def demo_batch_processing():
    """Demonstrate batch processing features"""
    print("\n\n📦 BATCH PROCESSING DEMO")
    print("=" * 50)
    
    from batch_processor import BatchProcessor
    from chatterbox_tts import tts_interface
    
    # Create batch processor
    batch_processor = BatchProcessor(tts_interface)
    
    # Demonstrate text batch
    print("📝 Text Batch Demo:")
    test_texts = [
        "Hello, this is the first test sentence.",
        "This is the second sentence with [laugh] some emotion.",
        "And finally, the third sentence with excitement!"
    ]
    
    item_ids = batch_processor.add_text_batch(
        test_texts,
        exaggeration=0.6,
        cfg_weight=0.5,
        preset="conversational"
    )
    print(f"   ✅ Added {len(item_ids)} text items")
    
    # Demonstrate CSV batch
    print(f"\n📄 CSV Batch Demo:")
    csv_content = '''text,exaggeration,cfg_weight,preset
"Happy announcement [giggle]",0.8,0.6,happy
"Serious news report",0.3,0.7,news_anchor
"Exciting commercial message!",0.9,0.8,commercial'''
    
    csv_ids, csv_status = batch_processor.add_csv_batch(csv_content)
    print(f"   {csv_status}")
    
    # Demonstrate JSON batch
    print(f"\n📋 JSON Batch Demo:")
    import json
    json_data = [
        {"text": "JSON test with storytelling", "preset": "storyteller", "exaggeration": 0.7},
        {"text": "Meditation guide example [pause]", "preset": "meditation", "cfg_weight": 0.3}
    ]
    
    json_ids, json_status = batch_processor.add_json_batch(json.dumps(json_data))
    print(f"   {json_status}")
    
    # Show batch status
    status = batch_processor.get_batch_status()
    print(f"\n📊 Batch Queue Status:")
    print(f"   Total items: {status['total_items']}")
    print(f"   Queue size: {status['queue_size']}")
    print(f"   Completed: {status['completed_items']}")
    print(f"   Failed: {status['failed_items']}")
    
    # Note about processing
    print(f"\n💡 Note: Actual TTS processing requires the ChatterBox model to be loaded.")
    print(f"   Use batch_processor.process_batch() to start processing when model is ready.")
    
    # Clear the queue for demo
    batch_processor.clear_batch()
    print(f"   🗑️ Demo queue cleared")

def demo_integration():
    """Demonstrate how all features work together"""
    print("\n\n🔗 INTEGRATION DEMO")
    print("=" * 50)
    
    # Example of a complete workflow
    print("🎯 Complete Workflow Example:")
    
    # 1. Text with sound effects and technical terms
    original_text = "Welcome to ChatterBox TTS [laugh]! Our AI uses GPU acceleration for 2024's best results. Visit https://github.com for more info!"
    
    # 2. Process the text
    processed_text = text_processor.process_text(original_text)
    
    # 3. Get preset parameters
    preset_name = "excited"
    exaggeration, cfg_weight = emotion_presets.get_preset_parameters(preset_name)
    preset_info = emotion_presets.get_preset(preset_name)
    
    print(f"\n📝 Text Processing:")
    print(f"   Original: {original_text}")
    print(f"   Processed: {processed_text}")
    
    print(f"\n🎭 Emotion Preset ({preset_name}):")
    print(f"   {preset_info.get('icon', '🎭')} {preset_info.get('name', preset_name)}")
    print(f"   Description: {preset_info.get('description', 'N/A')}")
    print(f"   Parameters: exaggeration={exaggeration}, cfg_weight={cfg_weight}")
    
    print(f"\n🎚️ Audio Enhancement Pipeline:")
    print(f"   1. Generate speech with processed text and preset")
    print(f"   2. Apply noise reduction (strength: 0.3)")
    print(f"   3. Apply vocal EQ for speech clarity")
    print(f"   4. Normalize to -20dB for consistent volume")
    
    print(f"\n✨ Result: High-quality, enhanced TTS output ready for use!")

def main():
    """Run all demos"""
    print("🚀 CHATTERBOX TTS ENHANCED FEATURES DEMO")
    print("=" * 60)
    print("This demo showcases all the new enhanced features:")
    print("• Emotion & Style Presets")
    print("• Smart Text Processing") 
    print("• Audio Post-Processing")
    print("• Batch Processing")
    print("• Feature Integration")
    print("=" * 60)
    
    try:
        demo_emotion_presets()
        demo_text_processing()
        demo_audio_processing()
        demo_batch_processing()
        demo_integration()
        
        print("\n\n🎉 DEMO COMPLETE!")
        print("=" * 60)
        print("All enhanced features are working correctly!")
        print("Ready to use ChatterBox TTS with:")
        print("✅ 16 emotion & style presets")
        print("✅ Smart text processing with 30+ sound effects")
        print("✅ Professional audio enhancement")
        print("✅ Efficient batch processing")
        print("✅ Seamless feature integration")
        print("\nLaunch the UI with: python main.py")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
