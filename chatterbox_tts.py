"""
ChatterBox TTS - ResembleAI Text-to-Speech Model Interface
Enhanced with audio post-processing, emotion presets, and batch processing
"""
import os
import tempfile
import numpy as np
import soundfile as sf
from typing import Optional, Tuple
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import new enhancement modules
try:
    from audio_processor import audio_processor
    from emotion_presets import emotion_presets
    from text_processor import text_processor
    ENHANCEMENTS_AVAILABLE = True
    print("✅ Enhancement modules loaded")
except ImportError as e:
    print(f"Warning: Enhancement modules not available: {e}")
    ENHANCEMENTS_AVAILABLE = False

try:
    import torch
    import torchaudio
    from chatterbox.tts import ChatterboxTTS
    TTS_AVAILABLE = True
    print(f"✅ ChatterBox TTS available")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
except ImportError as e:
    print(f"Warning: TTS dependencies not available: {e}")
    TTS_AVAILABLE = False

class ChatterBoxTTSInterface:
    """Interface for ResembleAI ChatterBox TTS model"""
    
    def __init__(self, auto_load=False):
        self.model = None
        if TTS_AVAILABLE:
            # Force CUDA if available (for NVIDIA GTX 1650)
            if torch.cuda.is_available():
                self.device = "cuda"
                print(f"🎮 Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                self.device = "cpu"
                print("💻 Using CPU (GPU not available)")
        else:
            self.device = "cpu"
        self.sample_rate = 24000  # Default sample rate for ChatterBox

        # Auto-load model if requested and dependencies are available
        if auto_load and TTS_AVAILABLE:
            print("🤖 Auto-loading ChatterBox TTS model...")
            self.load_model()
        
    def load_model(self) -> bool:
        """Load the ChatterBox TTS model"""
        if not TTS_AVAILABLE:
            print("❌ PyTorch and ChatterBox TTS dependencies not available")
            return False

        try:
            print(f"🤖 Loading ChatterBox TTS model on {self.device.upper()}...")
            if self.device == "cuda":
                print("   🎮 Using NVIDIA GTX 1650 GPU for acceleration")
            print("   📥 Loading model weights (may take a moment)...")

            # Load the model with explicit device
            self.model = ChatterboxTTS.from_pretrained(device=self.device)
            self.sample_rate = self.model.sr

            print("✅ Model loaded successfully!")
            print(f"   🎯 Device: {self.device.upper()}")
            print(f"   🎵 Sample Rate: {self.sample_rate} Hz")
            print(f"   🚀 Ready for real TTS generation!")
            return True

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("💡 Make sure you have installed:")
            print("   pip install torch torchaudio")
            print("   pip install chatterbox-tts")
            return False
    
    def generate_speech(
        self,
        text: str,
        audio_prompt_path: Optional[str] = None,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        seed: Optional[int] = None,
        preset: Optional[str] = None,
        enable_text_processing: bool = True,
        enable_audio_enhancement: bool = False,
        noise_reduction: float = 0.3,
        eq_preset: str = "vocal"
    ) -> Tuple[Optional[str], str]:
        """
        Generate speech from text using ChatterBox TTS with enhancements

        Args:
            text: Text to synthesize
            audio_prompt_path: Path to reference audio for voice cloning
            exaggeration: Emotion exaggeration level (0.0 to 1.0)
            cfg_weight: CFG weight for generation control (0.0 to 1.0)
            seed: Random seed for reproducible generation
            preset: Emotion/style preset to apply
            enable_text_processing: Whether to apply text preprocessing
            enable_audio_enhancement: Whether to apply audio post-processing
            noise_reduction: Noise reduction strength (0.0 to 1.0)
            eq_preset: EQ preset for audio enhancement

        Returns:
            Tuple of (audio_file_path, status_message)
        """
        if not self.model:
            return None, "❌ Model not loaded. Please load the model first."

        if not text.strip():
            return None, "❌ Please enter some text to synthesize."

        try:
            # Apply emotion/style preset if specified
            if preset and ENHANCEMENTS_AVAILABLE:
                preset_exaggeration, preset_cfg_weight = emotion_presets.get_preset_parameters(preset)
                exaggeration = preset_exaggeration
                cfg_weight = preset_cfg_weight
                print(f"🎭 Applied preset '{preset}': exaggeration={exaggeration:.1f}, cfg_weight={cfg_weight:.1f}")

            # Apply text processing if enabled
            processed_text = text
            if enable_text_processing and ENHANCEMENTS_AVAILABLE:
                processed_text = text_processor.process_text(text)
                if processed_text != text:
                    print(f"📝 Text processed: '{text[:30]}...' → '{processed_text[:30]}...'")
            else:
                processed_text = text
            # Set seed for reproducible generation
            if seed is not None and TTS_AVAILABLE:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)
                np.random.seed(seed)
            
            print(f"🎵 Generating speech for: '{processed_text[:50]}{'...' if len(processed_text) > 50 else ''}'")

            # Generate audio
            if audio_prompt_path and os.path.exists(audio_prompt_path):
                print(f"🎤 Using voice reference: {audio_prompt_path}")
                wav = self.model.generate(
                    processed_text,
                    audio_prompt_path=audio_prompt_path,
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight
                )
            else:
                wav = self.model.generate(
                    processed_text,
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight
                )
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                output_path = tmp_file.name
            
            # Convert to numpy array if it's a tensor
            if TTS_AVAILABLE and torch.is_tensor(wav):
                wav_np = wav.cpu().numpy()
            else:
                wav_np = wav
            
            # Ensure the audio is in the right format
            if wav_np.ndim > 1:
                wav_np = wav_np.squeeze()
            
            # Save audio file
            sf.write(output_path, wav_np, self.sample_rate)

            # Apply audio enhancement if enabled
            final_output_path = output_path
            enhancement_info = ""

            if enable_audio_enhancement and ENHANCEMENTS_AVAILABLE:
                print("🎚️ Applying audio enhancement...")
                enhanced_path, enhancement_msg = audio_processor.enhance_audio(
                    output_path,
                    noise_reduction=noise_reduction,
                    normalize=True,
                    target_db=-20.0,
                    eq_preset=eq_preset,
                    output_format="wav"
                )

                if enhanced_path:
                    final_output_path = enhanced_path
                    enhancement_info = f" + Enhanced with {eq_preset} EQ"
                    # Clean up original file
                    try:
                        os.unlink(output_path)
                    except:
                        pass

            duration = len(wav_np) / self.sample_rate
            status = f"✅ Generated {duration:.2f}s of audio"

            if audio_prompt_path:
                status += f" (voice cloned from reference)"

            if enhancement_info:
                status += enhancement_info

            return final_output_path, status
            
        except Exception as e:
            error_msg = f"❌ Error generating speech: {str(e)}"
            print(error_msg)
            return None, error_msg
    
    def get_model_info(self) -> str:
        """Get information about the loaded model"""
        if not self.model:
            return "❌ Model not loaded"
        
        info = f"""
        🤖 **ChatterBox TTS Model Info**
        
        - **Device**: {self.device}
        - **Sample Rate**: {self.sample_rate} Hz
        - **Model Type**: ResembleAI ChatterBox TTS
        - **Features**: Zero-shot TTS, Voice Cloning, Emotion Control
        - **Backbone**: 0.5B Llama-based architecture
        """
        return info
    
    def validate_audio_file(self, audio_path: str) -> Tuple[bool, str]:
        """Validate if an audio file can be used as reference"""
        if not audio_path or not os.path.exists(audio_path):
            return False, "❌ Audio file not found"
        
        try:
            # Try to load the audio file
            import librosa
            audio, sr = librosa.load(audio_path, sr=None)
            duration = len(audio) / sr
            
            if duration < 1.0:
                return False, "❌ Audio file too short (minimum 1 second)"
            elif duration > 30.0:
                return False, "⚠️ Audio file very long (>30s), may affect quality"
            else:
                return True, f"✅ Valid audio file ({duration:.1f}s, {sr}Hz)"
                
        except Exception as e:
            return False, f"❌ Error reading audio file: {str(e)}"

# Global instance - manual loading for better control
tts_interface = ChatterBoxTTSInterface(auto_load=False)
