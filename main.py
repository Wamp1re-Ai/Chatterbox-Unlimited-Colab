"""
ChatterBox TTS - Main application entry point
"""
import argparse
import sys
import os
from gradio_ui import ChatterBoxTTSUI

def main():
    """Main function to launch ChatterBox TTS"""
    parser = argparse.ArgumentParser(description="ChatterBox TTS - ResembleAI Text-to-Speech")
    parser.add_argument(
        "--share", 
        action="store_true", 
        help="Create a public shareable link"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=7860, 
        help="Port to run the server on (default: 7860)"
    )
    parser.add_argument(
        "--host", 
        type=str, 
        default="127.0.0.1", 
        help="Host to run the server on (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--debug", 
        action="store_true", 
        help="Enable debug mode"
    )
    
    args = parser.parse_args()
    
    # Print welcome message
    print("=" * 70)
    print("🎤 Welcome to ChatterBox TTS by ResembleAI!")
    print("   State-of-the-Art Text-to-Speech with Voice Cloning")
    print("=" * 70)
    print()
    
    # Check system requirements
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🖥️  Device: {device.upper()}")
        if device == "cuda":
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print()
    except ImportError:
        print("⚠️  PyTorch not found. Please install PyTorch first.")
        print("   Visit: https://pytorch.org/get-started/locally/")
        print()
    
    # Check if chatterbox-tts is available
    try:
        from chatterbox.tts import ChatterboxTTS
        print("✅ ChatterBox TTS package found")
    except ImportError:
        print("❌ ChatterBox TTS package not found")
        print("   Install with: pip install chatterbox-tts")
        print("   Or visit: https://github.com/resemble-ai/chatterbox")
        print()
        print("🔄 Continuing anyway - you can install dependencies later...")
        print()
    
    try:
        # Create and launch the UI
        ui = ChatterBoxTTSUI()
        ui.launch(
            share=args.share,
            server_port=args.port,
            server_name=args.host,
            debug=args.debug
        )
    except KeyboardInterrupt:
        print("\n👋 Thanks for using ChatterBox TTS! Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error launching ChatterBox TTS: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
