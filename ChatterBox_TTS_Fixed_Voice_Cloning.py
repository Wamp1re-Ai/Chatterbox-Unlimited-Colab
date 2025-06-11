#!/usr/bin/env python3
"""
ChatterBox TTS - Fixed Voice Cloning and Parallel Processing
Addresses infinite loop issues with voice cloning and improves batch processing
"""

import os
import sys
import torch
import torchaudio
import tempfile
import librosa
import soundfile as sf
import numpy as np
import threading
import time
import concurrent.futures
from functools import wraps
from chatterbox.tts import ChatterboxTTS

class TimeoutError(Exception):
    """Custom timeout exception"""
    pass

def with_timeout(timeout_seconds):
    """Decorator to add timeout to any function"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = [None]
            exception = [None]
            
            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    exception[0] = e
            
            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(timeout_seconds)
            
            if thread.is_alive():
                print(f"⏰ Operation timed out after {timeout_seconds} seconds")
                raise TimeoutError(f"Operation timed out after {timeout_seconds} seconds")
            
            if exception[0]:
                raise exception[0]
            
            return result[0]
        return wrapper
    return decorator

def clear_cuda_cache():
    """Clear CUDA cache and synchronize"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def preprocess_audio(audio_file):
    """Preprocess audio file for voice cloning"""
    if not audio_file or not os.path.exists(audio_file):
        return None, "❌ Audio file not found"
    
    try:
        print(f"🎵 Preprocessing audio: {audio_file}")
        
        # Load audio with librosa
        audio, sr = librosa.load(audio_file, sr=22050, mono=True)
        
        # Trim silence
        audio, _ = librosa.effects.trim(audio, top_db=20)
        
        # Normalize
        audio = librosa.util.normalize(audio)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            sf.write(tmp_file.name, audio, sr)
            processed_path = tmp_file.name
        
        duration = len(audio) / sr
        print(f"✅ Audio preprocessed: {duration:.1f}s, {sr}Hz")
        
        return processed_path, f"✅ Audio preprocessed: {duration:.1f}s"
        
    except Exception as e:
        return None, f"❌ Audio preprocessing failed: {str(e)}"

def smart_text_chunking(text, max_chunk_size=200):
    """Split text into chunks at natural boundaries"""
    if len(text) <= max_chunk_size:
        return [text]
    
    # Split by sentences first
    import re
    sentences = re.split(r'[.!?]+', text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # If adding this sentence would exceed the limit
        if len(current_chunk) + len(sentence) + 1 > max_chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                # Single sentence is too long, split by words
                words = sentence.split()
                temp_chunk = ""
                for word in words:
                    if len(temp_chunk) + len(word) + 1 <= max_chunk_size:
                        temp_chunk += " " + word if temp_chunk else word
                    else:
                        if temp_chunk:
                            chunks.append(temp_chunk)
                        temp_chunk = word
                if temp_chunk:
                    current_chunk = temp_chunk
        else:
            current_chunk += ". " + sentence if current_chunk else sentence
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

@with_timeout(60)  # 60 second timeout per chunk
def generate_chunk_with_timeout(model, chunk_text, processed_audio_path=None, exaggeration=0.5, cfg_weight=0.5):
    """Generate a single chunk with timeout protection"""
    clear_cuda_cache()
    
    if processed_audio_path is not None:
        # Voice cloning mode
        return model.generate(
            chunk_text, 
            audio_prompt_path=processed_audio_path,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight
        )
    else:
        # Standard TTS mode
        return model.generate(
            chunk_text,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight
        )

def generate_speech_fixed(model, text, audio_file=None, exaggeration=0.5, cfg_weight=0.5, speed_factor=1.0):
    """
    Fixed speech generation with proper voice cloning and parallel processing
    """
    try:
        print(f"🎤 Starting speech generation...")
        print(f"📝 Text length: {len(text)} characters")
        
        # Store original text for reporting
        original_text = text
        
        # Preprocess audio if provided
        processed_audio_path = None
        if audio_file is not None:
            processed_audio_path, preprocess_msg = preprocess_audio(audio_file)
            if processed_audio_path is None:
                return None, preprocess_msg
            print(preprocess_msg)
        
        # Smart text chunking
        chunks = smart_text_chunking(text, max_chunk_size=200)
        total_chunks = len(chunks)
        print(f"📊 Split into {total_chunks} chunks")
        
        # Decide processing strategy based on voice cloning
        use_sequential = processed_audio_path is not None  # Use sequential for voice cloning
        
        if use_sequential:
            print(f"🎭 Voice cloning detected - using SEQUENTIAL processing for stability...")
            print(f"📝 Processing {total_chunks} chunks one by one to avoid CUDA conflicts")
        else:
            print(f"🚀 Using PARALLEL processing for standard TTS...")
        
        all_audio_chunks = []
        total_duration = 0
        
        if use_sequential:
            # Sequential processing for voice cloning
            for i, chunk in enumerate(chunks):
                try:
                    print(f"\n🎤 Processing chunk {i + 1}/{total_chunks} sequentially...")
                    print(f"📝 Chunk text: '{chunk[:50]}...'")
                    
                    chunk_wav = generate_chunk_with_timeout(
                        model, chunk, processed_audio_path, exaggeration, cfg_weight
                    )
                    
                    all_audio_chunks.append(chunk_wav)
                    chunk_duration = chunk_wav.shape[1] / model.sr
                    total_duration += chunk_duration
                    print(f"✅ Chunk {i + 1}/{total_chunks} completed: {chunk_duration:.1f}s")
                    
                except TimeoutError as e:
                    print(f"⏰ Chunk {i + 1} timed out: {str(e)}")
                    raise e
                except Exception as e:
                    print(f"❌ Chunk {i + 1} failed: {str(e)}")
                    raise e
            
            print(f"🎉 All {total_chunks} chunks completed sequentially!")
        
        else:
            # Parallel processing for standard TTS (limited workers)
            max_workers = min(2, total_chunks)  # Limit to 2 workers
            
            def generate_chunk_wrapper(chunk_data):
                chunk_idx, chunk_text = chunk_data
                print(f"\n🎤 [Worker {chunk_idx + 1}] Processing chunk {chunk_idx + 1}/{total_chunks}")
                
                try:
                    chunk_wav = generate_chunk_with_timeout(
                        model, chunk_text, None, exaggeration, cfg_weight
                    )
                    chunk_duration = chunk_wav.shape[1] / model.sr
                    print(f"✅ [Worker {chunk_idx + 1}] Completed: {chunk_duration:.1f}s")
                    return chunk_idx, chunk_wav, chunk_duration
                except Exception as e:
                    print(f"❌ [Worker {chunk_idx + 1}] Failed: {str(e)}")
                    raise e
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all chunk generation tasks
                future_to_chunk = {executor.submit(generate_chunk_wrapper, (i, chunk)): i for i, chunk in enumerate(chunks)}
                
                # Pre-allocate list to maintain order
                ordered_chunks = [None] * total_chunks
                
                # Collect results as they complete with timeout
                try:
                    for future in concurrent.futures.as_completed(future_to_chunk, timeout=300):  # 5 minute total timeout
                        chunk_idx, chunk_wav, chunk_duration = future.result(timeout=60)  # 60 second per chunk timeout
                        ordered_chunks[chunk_idx] = chunk_wav
                        total_duration += chunk_duration
                        print(f"📦 Collected chunk {chunk_idx + 1}/{total_chunks}")
                        
                except concurrent.futures.TimeoutError:
                    print("⏰ Parallel processing timed out - this shouldn't happen with sequential voice cloning")
                    raise TimeoutError("Parallel processing timed out")
                
                all_audio_chunks = ordered_chunks
            
            print(f"🎉 All {total_chunks} chunks completed in parallel!")
        
        # Concatenate all audio chunks
        if len(all_audio_chunks) == 1:
            final_wav = all_audio_chunks[0]
        else:
            print(f"🔗 Concatenating {len(all_audio_chunks)} audio chunks...")
            final_wav = torch.cat(all_audio_chunks, dim=1)
        
        # Apply speed adjustment if needed
        if speed_factor != 1.0:
            print(f"🎵 Adjusting speech speed by {speed_factor}x...")
            wav_np = final_wav.cpu().numpy().squeeze()
            wav_stretched = librosa.effects.time_stretch(wav_np, rate=speed_factor)
            final_wav = torch.from_numpy(wav_stretched).unsqueeze(0)
            total_duration = total_duration / speed_factor
        
        # Save final audio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            torchaudio.save(tmp_file.name, final_wav, model.sr)
            output_path = tmp_file.name
        
        # Clean up preprocessed audio file
        if processed_audio_path and os.path.exists(processed_audio_path):
            try:
                os.unlink(processed_audio_path)
            except:
                pass
        
        # Create success message
        success_msg = f"✅ Generated {total_duration:.1f}s of audio from {len(original_text)} characters"
        if total_chunks > 1:
            success_msg += f" (processed in {total_chunks} chunks)"
        if audio_file is not None:
            success_msg += " (voice cloned)"
        if use_sequential:
            success_msg += " [SEQUENTIAL MODE]"
        else:
            success_msg += " [PARALLEL MODE]"
        
        print(success_msg)
        return output_path, success_msg
        
    except Exception as e:
        error_msg = f"❌ Generation failed: {str(e)}"
        print(error_msg)
        
        # Clean up on error
        if 'processed_audio_path' in locals() and processed_audio_path and os.path.exists(processed_audio_path):
            try:
                os.unlink(processed_audio_path)
            except:
                pass
        
        return None, error_msg

if __name__ == "__main__":
    print("🎤 ChatterBox TTS - Fixed Voice Cloning Implementation")
    print("=" * 60)
    print("✅ Sequential processing for voice cloning")
    print("✅ Timeout protection (60s per chunk)")
    print("✅ Improved error handling")
    print("✅ Smart text chunking")
    print("=" * 60)
