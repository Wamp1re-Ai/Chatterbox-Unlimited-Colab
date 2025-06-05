"""
Gradio UI for ChatterBox TTS
"""
import gradio as gr
import os
from typing import Tuple, Optional
from chatterbox_tts import tts_interface

class ChatterBoxTTSUI:
    """Gradio UI wrapper for ChatterBox TTS"""
    
    def __init__(self):
        self.interface = tts_interface
        self.model_loaded = self.interface.model is not None
    
    def load_model_ui(self) -> str:
        """Load the TTS model"""
        if self.interface.load_model():
            self.model_loaded = True
            return "✅ ChatterBox TTS model loaded successfully!"
        else:
            return "❌ Failed to load model. Please check dependencies."
    
    def generate_speech_ui(
        self,
        text: str,
        reference_audio,
        exaggeration: float,
        cfg_weight: float,
        seed: Optional[int]
    ) -> Tuple[Optional[str], str]:
        """Generate speech with UI wrapper"""
        if not self.model_loaded:
            return None, "❌ Please load the model first."
        
        # Handle reference audio
        audio_path = None
        if reference_audio is not None:
            audio_path = reference_audio
        
        return self.interface.generate_speech(
            text=text,
            audio_prompt_path=audio_path,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            seed=seed
        )
    
    def validate_reference_audio(self, audio_file) -> str:
        """Validate reference audio file"""
        if audio_file is None:
            return "ℹ️ No reference audio provided - will use default voice"
        
        is_valid, message = self.interface.validate_audio_file(audio_file)
        return message
    
    def create_interface(self) -> gr.Blocks:
        """Create the main Gradio interface"""
        
        # Custom CSS for better styling
        custom_css = """
        .main-header {
            text-align: center;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        
        .control-panel {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #dee2e6;
        }
        
        .info-box {
            background: #e3f2fd;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #2196f3;
        }
        
        .warning-box {
            background: #fff3e0;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #ff9800;
        }
        """
        
        with gr.Blocks(
            theme=gr.themes.Soft(),
            css=custom_css,
            title="ChatterBox TTS - ResembleAI"
        ) as interface:
            
            # Header
            gr.HTML("""
            <div class="main-header">
                <h1>🎤 ChatterBox TTS</h1>
                <p>ResembleAI's State-of-the-Art Text-to-Speech Model</p>
                <p><em>Zero-shot TTS • Voice Cloning • Emotion Control</em></p>
            </div>
            """)
            
            # Model loading section
            with gr.Row():
                with gr.Column():
                    gr.Markdown("## 🚀 Model Setup")
                    load_btn = gr.Button("Load ChatterBox TTS Model", variant="primary", size="lg")
                    initial_status = "✅ Model loaded successfully!" if self.model_loaded else "❌ Model not loaded. Click 'Load Model' to start."
                    model_status = gr.Textbox(
                        label="Model Status",
                        value=initial_status,
                        interactive=False
                    )
            
            gr.Markdown("---")
            
            # Main TTS interface
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("## 📝 Text Input")
                    text_input = gr.Textbox(
                        label="Text to Synthesize",
                        placeholder="Enter the text you want to convert to speech...",
                        lines=4,
                        max_lines=10
                    )
                    
                    # Example texts
                    gr.Examples(
                        examples=[
                            ["Hello! Welcome to ChatterBox TTS by ResembleAI."],
                            ["The quick brown fox jumps over the lazy dog."],
                            ["In a hole in the ground there lived a hobbit."],
                            ["To be or not to be, that is the question."],
                            ["The future belongs to those who believe in the beauty of their dreams."]
                        ],
                        inputs=[text_input],
                        label="Example Texts"
                    )
                
                with gr.Column(scale=1):
                    gr.Markdown("## 🎛️ Controls")
                    
                    with gr.Group(elem_classes=["control-panel"]):
                        exaggeration = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.5,
                            step=0.1,
                            label="Emotion Exaggeration"
                        )
                        gr.Markdown("*Higher values = more expressive speech*")

                        cfg_weight = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.5,
                            step=0.1,
                            label="CFG Weight"
                        )
                        gr.Markdown("*Lower values = slower, more deliberate pacing*")

                        seed_input = gr.Number(
                            label="Seed (optional)",
                            value=None,
                            precision=0
                        )
                        gr.Markdown("*For reproducible generation*")
            
            # Voice cloning section
            gr.Markdown("## 🎭 Voice Cloning (Optional)")
            with gr.Row():
                with gr.Column():
                    reference_audio = gr.Audio(
                        label="Reference Audio",
                        type="filepath"
                    )
                    gr.Markdown("*Upload an audio file to clone the voice (1-30 seconds recommended)*")
                    audio_status = gr.Textbox(
                        label="Audio Validation",
                        value="ℹ️ No reference audio provided - will use default voice",
                        interactive=False
                    )
            
            # Generation section
            gr.Markdown("## 🎵 Generate Speech")
            with gr.Row():
                generate_btn = gr.Button("🎤 Generate Speech", variant="primary", size="lg")
            
            with gr.Row():
                with gr.Column():
                    output_audio = gr.Audio(
                        label="Generated Speech",
                        type="filepath"
                    )
                    generation_status = gr.Textbox(
                        label="Generation Status",
                        interactive=False
                    )
            
            # Tips and information
            gr.Markdown("---")
            with gr.Accordion("💡 Tips & Information", open=False):
                gr.HTML("""
                <div class="info-box">
                    <h3>🎯 Usage Tips</h3>
                    <ul>
                        <li><strong>General Use:</strong> Default settings (exaggeration=0.5, cfg=0.5) work well for most texts</li>
                        <li><strong>Fast Speakers:</strong> Lower CFG weight to ~0.3 for better pacing</li>
                        <li><strong>Expressive Speech:</strong> Lower CFG (~0.3) + higher exaggeration (~0.7+)</li>
                        <li><strong>Voice Cloning:</strong> Use clear, 3-10 second audio samples for best results</li>
                    </ul>
                </div>

                <div class="warning-box">
                    <h3>⚠️ Important Notes</h3>
                    <ul>
                        <li>Generated audio includes watermarking for responsible AI use</li>
                        <li>First generation may take longer due to model initialization</li>
                        <li>GPU recommended for faster generation</li>
                        <li>Use responsibly and respect voice ownership rights</li>
                    </ul>
                </div>
                """)

            # Event handlers
            load_btn.click(
                fn=self.load_model_ui,
                outputs=[model_status]
            )

            reference_audio.change(
                fn=self.validate_reference_audio,
                inputs=[reference_audio],
                outputs=[audio_status]
            )

            generate_btn.click(
                fn=self.generate_speech_ui,
                inputs=[text_input, reference_audio, exaggeration, cfg_weight, seed_input],
                outputs=[output_audio, generation_status]
            )
        
        return interface
    
    def launch(self, **kwargs):
        """Launch the Gradio interface"""
        interface = self.create_interface()
        
        # Default launch parameters
        launch_params = {
            "share": False,
            "server_name": "127.0.0.1",
            "server_port": 7860,
            "show_error": True,
            "quiet": False
        }
        
        # Update with user-provided parameters
        launch_params.update(kwargs)
        
        print("🚀 Launching ChatterBox TTS UI...")
        print(f"🌐 Access the interface at: http://{launch_params['server_name']}:{launch_params['server_port']}")
        print("🎤 ResembleAI ChatterBox TTS - State-of-the-Art Text-to-Speech")
        
        interface.launch(**launch_params)
