# 🎤 ChatterBox Unlimited - Colab Ready TTS

A powerful Gradio web interface for **ResembleAI's ChatterBox TTS** with Google Colab support. Generate unlimited high-quality speech with voice cloning capabilities!

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Wamp1re-Ai/Chatterbox-Unlimited-Colab/blob/main/ChatterBox_Unlimited_Colab.ipynb)

## ✨ Features

- **🎯 Zero-shot TTS**: Generate speech from any text without training
- **🎭 Voice Cloning**: Clone voices from reference audio samples
- **🎨 Emotion Control**: Adjust emotion exaggeration for expressive speech
- **⚙️ Fine-tuned Control**: CFG weight adjustment for pacing control
- **🌐 Web Interface**: Beautiful, responsive Gradio UI
- **🔄 Reproducible**: Seed control for consistent generation
- **🎵 High Quality**: 24kHz audio output with watermarking
- **🚀 GPU Acceleration**: CUDA support for faster generation
- **☁️ Colab Ready**: One-click deployment on Google Colab

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

1. **"Model not found"**: Run `python install_dependencies.py`
2. **CUDA errors**: Install compatible PyTorch version
3. **Memory errors**: Try CPU mode or close other applications
4. **Audio issues**: Check file format and quality

## 🔗 Links

- [ChatterBox TTS GitHub](https://github.com/resemble-ai/chatterbox)
- [ResembleAI](https://www.resemble.ai/)
- [Hugging Face Model](https://huggingface.co/ResembleAI/chatterbox)
- [Official Demo](https://huggingface.co/spaces/ResembleAI/Chatterbox)

---

**Enjoy creating amazing speech with ChatterBox TTS! 🎉**
