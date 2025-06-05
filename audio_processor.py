"""
Audio Post-Processing Module for ChatterBox TTS
Handles noise reduction, normalization, format conversion, and quality enhancement
"""
import os
import tempfile
import numpy as np
import soundfile as sf
from typing import Optional, Tuple, Dict, Any
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

try:
    import librosa
    import scipy.signal
    from scipy.io import wavfile
    AUDIO_PROCESSING_AVAILABLE = True
    print("✅ Audio processing libraries available")
except ImportError as e:
    print(f"Warning: Audio processing dependencies not available: {e}")
    AUDIO_PROCESSING_AVAILABLE = False

class AudioProcessor:
    """Advanced audio post-processing for TTS output"""
    
    def __init__(self):
        self.sample_rate = 24000  # Default ChatterBox sample rate
        
    def reduce_noise(self, audio: np.ndarray, sr: int, noise_reduction_strength: float = 0.5) -> np.ndarray:
        """
        Apply noise reduction using spectral gating
        
        Args:
            audio: Audio signal
            sr: Sample rate
            noise_reduction_strength: Strength of noise reduction (0.0 to 1.0)
            
        Returns:
            Processed audio with reduced noise
        """
        if not AUDIO_PROCESSING_AVAILABLE:
            return audio
            
        try:
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            # Compute STFT
            stft = librosa.stft(audio, n_fft=2048, hop_length=512)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Estimate noise floor from first 0.5 seconds
            noise_frames = int(0.5 * sr / 512)  # 512 is hop_length
            noise_profile = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
            
            # Apply spectral gating
            noise_gate_threshold = noise_profile * (2.0 + noise_reduction_strength * 3.0)
            mask = magnitude > noise_gate_threshold
            
            # Smooth the mask to avoid artifacts
            mask = scipy.signal.medfilt(mask.astype(float), kernel_size=(1, 3))
            
            # Apply mask with soft gating
            processed_magnitude = magnitude * (mask * (1 - noise_reduction_strength) + noise_reduction_strength)
            
            # Reconstruct audio
            processed_stft = processed_magnitude * np.exp(1j * phase)
            processed_audio = librosa.istft(processed_stft, hop_length=512)
            
            return processed_audio
            
        except Exception as e:
            print(f"Warning: Noise reduction failed: {e}")
            return audio
    
    def normalize_audio(self, audio: np.ndarray, target_db: float = -20.0) -> np.ndarray:
        """
        Normalize audio to target dB level
        
        Args:
            audio: Audio signal
            target_db: Target dB level
            
        Returns:
            Normalized audio
        """
        try:
            # Calculate RMS
            rms = np.sqrt(np.mean(audio**2))
            
            if rms > 0:
                # Convert target dB to linear scale
                target_rms = 10**(target_db / 20.0)
                
                # Apply normalization
                normalized_audio = audio * (target_rms / rms)
                
                # Prevent clipping
                max_val = np.max(np.abs(normalized_audio))
                if max_val > 0.95:
                    normalized_audio = normalized_audio * (0.95 / max_val)
                
                return normalized_audio
            else:
                return audio
                
        except Exception as e:
            print(f"Warning: Normalization failed: {e}")
            return audio
    
    def apply_eq(self, audio: np.ndarray, sr: int, eq_preset: str = "balanced") -> np.ndarray:
        """
        Apply equalization to enhance audio quality
        
        Args:
            audio: Audio signal
            sr: Sample rate
            eq_preset: EQ preset ("balanced", "warm", "bright", "vocal")
            
        Returns:
            Equalized audio
        """
        if not AUDIO_PROCESSING_AVAILABLE:
            return audio
            
        try:
            # EQ presets (frequency, gain in dB, Q factor)
            eq_presets = {
                "balanced": [(100, 1.0, 0.7), (1000, 0.5, 1.0), (3000, 1.0, 0.8), (8000, 0.5, 0.7)],
                "warm": [(200, 2.0, 0.8), (800, 1.0, 1.0), (3000, -0.5, 0.8), (8000, -1.0, 0.7)],
                "bright": [(100, -0.5, 0.7), (1000, 0.0, 1.0), (3000, 1.5, 0.8), (8000, 2.0, 0.7)],
                "vocal": [(150, 1.0, 0.8), (1000, 1.5, 1.2), (2500, 2.0, 1.0), (5000, 1.0, 0.8)]
            }
            
            if eq_preset not in eq_presets:
                return audio
            
            processed_audio = audio.copy()
            
            for freq, gain, q in eq_presets[eq_preset]:
                # Design peaking filter
                nyquist = sr / 2
                normalized_freq = freq / nyquist
                
                if 0 < normalized_freq < 1:
                    b, a = scipy.signal.iirpeak(normalized_freq, Q=q)
                    
                    # Apply gain
                    if gain != 0:
                        gain_linear = 10**(gain / 20.0)
                        b = b * gain_linear
                    
                    # Apply filter
                    processed_audio = scipy.signal.filtfilt(b, a, processed_audio)
            
            return processed_audio
            
        except Exception as e:
            print(f"Warning: EQ processing failed: {e}")
            return audio
    
    def enhance_audio(
        self, 
        audio_path: str, 
        noise_reduction: float = 0.3,
        normalize: bool = True,
        target_db: float = -20.0,
        eq_preset: str = "vocal",
        output_format: str = "wav"
    ) -> Tuple[Optional[str], str]:
        """
        Apply comprehensive audio enhancement
        
        Args:
            audio_path: Path to input audio file
            noise_reduction: Noise reduction strength (0.0 to 1.0)
            normalize: Whether to normalize audio
            target_db: Target dB level for normalization
            eq_preset: EQ preset to apply
            output_format: Output format (wav, mp3, flac, ogg)
            
        Returns:
            Tuple of (enhanced_audio_path, status_message)
        """
        if not os.path.exists(audio_path):
            return None, "❌ Input audio file not found"
        
        try:
            # Load audio
            audio, sr = sf.read(audio_path)
            
            print(f"🎵 Enhancing audio: {os.path.basename(audio_path)}")
            print(f"   📊 Original: {len(audio)} samples, {sr} Hz")
            
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            # Apply noise reduction
            if noise_reduction > 0:
                print(f"   🔇 Applying noise reduction (strength: {noise_reduction:.1f})")
                audio = self.reduce_noise(audio, sr, noise_reduction)
            
            # Apply EQ
            if eq_preset and eq_preset != "none":
                print(f"   🎚️ Applying EQ preset: {eq_preset}")
                audio = self.apply_eq(audio, sr, eq_preset)
            
            # Normalize audio
            if normalize:
                print(f"   📈 Normalizing to {target_db} dB")
                audio = self.normalize_audio(audio, target_db)
            
            # Save enhanced audio
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{output_format}") as tmp_file:
                output_path = tmp_file.name
            
            # Save in requested format
            if output_format.lower() == "wav":
                sf.write(output_path, audio, sr)
            elif output_format.lower() == "mp3":
                # For MP3, we'll save as WAV and note the format request
                sf.write(output_path.replace('.mp3', '.wav'), audio, sr)
                output_path = output_path.replace('.mp3', '.wav')
                print("   ℹ️ MP3 encoding requires additional dependencies. Saved as WAV.")
            elif output_format.lower() == "flac":
                sf.write(output_path, audio, sr, format='FLAC')
            elif output_format.lower() == "ogg":
                sf.write(output_path, audio, sr, format='OGG')
            else:
                sf.write(output_path, audio, sr)
            
            enhancement_info = f"""
            ✅ Audio enhancement complete!
            
            🎵 **Processing Applied:**
            - Noise Reduction: {noise_reduction:.1f}
            - EQ Preset: {eq_preset}
            - Normalization: {target_db} dB
            - Output Format: {output_format.upper()}
            
            📊 **Audio Info:**
            - Duration: {len(audio) / sr:.2f} seconds
            - Sample Rate: {sr} Hz
            - Channels: Mono
            """
            
            return output_path, enhancement_info
            
        except Exception as e:
            return None, f"❌ Audio enhancement failed: {str(e)}"
    
    def convert_format(self, audio_path: str, output_format: str) -> Tuple[Optional[str], str]:
        """
        Convert audio to different format
        
        Args:
            audio_path: Path to input audio
            output_format: Target format (wav, flac, ogg)
            
        Returns:
            Tuple of (converted_audio_path, status_message)
        """
        try:
            audio, sr = sf.read(audio_path)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{output_format}") as tmp_file:
                output_path = tmp_file.name
            
            if output_format.lower() == "wav":
                sf.write(output_path, audio, sr)
            elif output_format.lower() == "flac":
                sf.write(output_path, audio, sr, format='FLAC')
            elif output_format.lower() == "ogg":
                sf.write(output_path, audio, sr, format='OGG')
            else:
                return None, f"❌ Unsupported format: {output_format}"
            
            return output_path, f"✅ Converted to {output_format.upper()}"
            
        except Exception as e:
            return None, f"❌ Format conversion failed: {str(e)}"

# Global audio processor instance
audio_processor = AudioProcessor()
