# 🎤 ChatterBox Unlimited - Colab Ready TTS

A powerful Gradio web interface for **ResembleAI's ChatterBox TTS** with Google Colab support. Generate unlimited high-quality speech with voice cloning capabilities!

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Wamp1re-Ai/Chatterbox-Unlimited-Colab/blob/main/ChatterBox_Unlimited_Colab.ipynb)
[![GitHub Stars](https://img.shields.io/github/stars/Wamp1re-Ai/Chatterbox-Unlimited-Colab?style=social)](https://github.com/Wamp1re-Ai/Chatterbox-Unlimited-Colab)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)

## ✨ Features

### 🎤 Core TTS Features
- **🎯 Zero-shot TTS**: Generate speech from any text without training
- **🎭 Voice Cloning**: Clone voices from reference audio samples
- **🎨 Emotion Control**: Adjust emotion exaggeration for expressive speech
- **⚙️ Fine-tuned Control**: CFG weight adjustment for pacing control
- **🔄 Reproducible**: Seed control for consistent generation
- **🎵 High Quality**: 24kHz audio output with watermarking
- **🚀 GPU Acceleration**: CUDA support for faster generation

### 🆕 Enhanced Features
- **🎭 Emotion & Style Presets**: Pre-configured emotion profiles (happy, sad, excited, calm, etc.) and speaking styles (news anchor, storyteller, commercial, etc.)
- **🎚️ Audio Post-Processing**: Advanced noise reduction, EQ presets (vocal, warm, bright), and audio normalization
- **📦 Batch Processing**: Process multiple texts at once with CSV/JSON import and bulk download
- **📝 Smart Text Processing**: Pronunciation dictionary, sound effects ([laugh], [cough], etc.), and text preprocessing
- **🌐 Enhanced Web Interface**: Tabbed interface with advanced controls and real-time feedback
- **☁️ Colab Ready**: One-click deployment on Google Colab with all enhancements

## 🚀 Quick Start

### 🌟 Option 1: Google Colab (Recommended)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Wamp1re-Ai/Chatterbox-Unlimited-Colab/blob/main/ChatterBox_Unlimited_Colab.ipynb)

1. Click the "Open in Colab" button above
2. Run all cells in the notebook
3. Access the Gradio interface through the provided link
4. Start generating speech with GPU acceleration!

### 💻 Option 2: Local Installation

#### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (recommended) or CPU
- 4GB+ RAM (8GB+ recommended)

### Installation

#### Option 1: Automatic Installation (Recommended)
```bash
python install_dependencies.py
```

#### Option 2: Manual Installation

1. **Install PyTorch** (visit [pytorch.org](https://pytorch.org/get-started/locally/)):
   ```bash
   # For CUDA (GPU support)
   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
   
   # For CPU only
   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

2. **Install ChatterBox TTS**:
   ```bash
   pip install chatterbox-tts
   ```

3. **Install other dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```bash
   python main.py
   ```

5. **Open your browser** and go to `http://127.0.0.1:7860`

## 🎛️ Usage Guide

### Basic Text-to-Speech

1. **Model Loading**: The model will auto-load if dependencies are installed, or click "Load ChatterBox TTS Model"
2. **Enter Text**: Type the text you want to synthesize
3. **Adjust Settings** (optional):
   - **Exaggeration**: 0.5 (default) - higher for more expressive speech
   - **CFG Weight**: 0.5 (default) - lower for slower pacing
4. **Generate**: Click "🎤 Generate Speech"

### Voice Cloning

1. **Upload Reference Audio**: 3-10 seconds of clear speech
2. **Follow Basic TTS steps** above
3. Generated speech will mimic the reference voice

## 🎯 Tips

- **General Use**: Default settings work well
- **Fast Speakers**: Lower CFG weight to ~0.3
- **Expressive Speech**: Lower CFG (~0.3) + higher exaggeration (~0.7+)
- **Voice Cloning**: Use clear, high-quality reference audio

## 📁 Project Structure

```
ChatterBox/
├── main.py                    # Application entry point
├── chatterbox_tts.py         # Core TTS functionality
├── gradio_ui.py              # Gradio web interface
├── install_dependencies.py   # Automatic installer
├── test_interface.py         # Testing script
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## 🛠️ Command Line Options

```bash
python main.py --help
```

Options:
- `--share`: Create public link
- `--port PORT`: Custom port (default: 7860)
- `--host HOST`: Custom host (default: 127.0.0.1)
- `--debug`: Enable debug mode

## 🆕 Enhanced Features Guide

### 🎭 Emotion & Style Presets

Choose from pre-configured emotion and style presets for consistent, high-quality results:

**Emotion Presets:**
- 😊 Happy - Cheerful, upbeat tone
- 😢 Sad - Melancholic, slower paced
- 🤩 Excited - High energy, enthusiastic
- 😌 Calm - Peaceful, soothing tone
- 😠 Angry - Intense, forceful delivery
- 😲 Surprised - Sudden, unexpected emphasis
- 😎 Confident - Strong, authoritative delivery

**Style Presets:**
- 💬 Conversational - Natural, casual conversation
- 📺 News Anchor - Professional, clear delivery
- 📚 Storyteller - Engaging, dramatic narrative
- 🧘 Meditation Guide - Slow, peaceful delivery
- 🎧 Audiobook Narrator - Clear, consistent reading
- 📊 Presentation - Professional, engaging style
- 📢 Commercial - Persuasive, energetic advertising

### 🎚️ Audio Post-Processing

Enhance your generated audio with professional-grade processing:

- **Noise Reduction**: Remove background noise and artifacts
- **EQ Presets**:
  - Vocal: Optimized for speech clarity
  - Warm: Rich, full-bodied sound
  - Bright: Clear, crisp highs
  - Balanced: Even frequency response
- **Normalization**: Consistent volume levels
- **Format Conversion**: WAV, FLAC, OGG support

### 📦 Batch Processing

Process multiple texts efficiently:

- **Text Lines**: Enter multiple lines, each processed separately
- **CSV Import**: Upload CSV with custom parameters per line
- **JSON Import**: Structured data with full control
- **Bulk Download**: Get all results in a single ZIP file
- **Progress Tracking**: Real-time status updates
- **Queue Management**: Add, start, stop, and clear processing queues

### 📝 Smart Text Processing

Automatic text enhancement for better TTS results:

- **Pronunciation Dictionary**: Automatic correction of common mispronunciations
- **Sound Effects**: Support for [laugh], [cough], [sigh], [pause], etc.
- **Number Processing**: Convert numbers to words (2024 → "twenty twenty-four")
- **URL/Email Handling**: Convert links to speakable format
- **Custom Pronunciations**: Add your own pronunciation rules

### 🎯 Quality Improvements

Based on original ChatterBox repository issues, we've implemented fixes for:

- **Audio Distortion**: Enhanced audio processing pipeline
- **Voice Quality**: Improved reference audio handling
- **Pronunciation Issues**: Comprehensive pronunciation dictionary
- **Sound Effects**: Support for non-speech sounds like [giggle], [cough]

## 🔧 Technical Details

- **Model**: ResembleAI ChatterBox TTS from Hugging Face
- **Architecture**: 0.5B parameter Llama-based backbone
- **Training Data**: 500K hours of cleaned audio
- **Sample Rate**: 24kHz
- **Features**: Zero-shot TTS, voice cloning, emotion control
- **Watermarking**: Built-in Perth watermarking

## ⚠️ Important Notes

- Generated audio includes watermarking for responsible AI use
- Respect voice ownership and consent when cloning
- First model load downloads ~2GB of weights
- GPU recommended for faster generation

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Version Conflicts** (Colab):
   ```
   RuntimeError: PyTorch and torchvision compiled with different CUDA versions
   ```
   **Solution**: Restart runtime and re-run all cells. The notebook now uses compatible versions.

2. **Dependency Conflicts**:
   ```
   ERROR: pip's dependency resolver conflicts
   ```
   **Solution**: Use the UV setup script:
   ```bash
   python setup_uv.py
   ```

3. **Model Loading Fails**:
   ```
   Failed to load ChatterBox TTS model
   ```
   **Solution**:
   - Ensure GPU has enough memory (4GB+ recommended)
   - Try CPU mode if GPU fails
   - Restart Python kernel/runtime

4. **Import Errors**:
   ```
   ImportError: cannot import ChatterboxTTS
   ```
   **Solution**: Install from GitHub:
   ```bash
   pip install git+https://github.com/resemble-ai/chatterbox.git
   ```

### Performance Tips

- **Colab**: Use GPU runtime for 10x faster generation
- **Local**: NVIDIA GPUs with 4GB+ VRAM work best
- **Memory**: Close other applications if running locally

## 🔗 Links

- [ChatterBox TTS GitHub](https://github.com/resemble-ai/chatterbox)
- [ResembleAI](https://www.resemble.ai/)
- [Hugging Face Model](https://huggingface.co/ResembleAI/chatterbox)
- [Official Demo](https://huggingface.co/spaces/ResembleAI/Chatterbox)

---

**Enjoy creating amazing speech with ChatterBox TTS! 🎉**
