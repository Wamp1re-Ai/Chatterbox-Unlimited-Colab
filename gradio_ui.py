"""
Gradio UI for ChatterBox TTS
Enhanced with audio post-processing, emotion presets, and batch processing
"""
import gradio as gr
import os
import json
import tempfile
from typing import Tuple, Optional, List
from chatterbox_tts import tts_interface

# Import enhancement modules
try:
    from emotion_presets import emotion_presets
    from text_processor import text_processor
    from batch_processor import BatchProcessor
    ENHANCEMENTS_AVAILABLE = True
except ImportError:
    ENHANCEMENTS_AVAILABLE = False

class ChatterBoxTTSUI:
    """Enhanced Gradio UI wrapper for ChatterBox TTS"""

    def __init__(self):
        self.interface = tts_interface
        self.model_loaded = self.interface.model is not None

        # Initialize batch processor if enhancements are available
        if ENHANCEMENTS_AVAILABLE:
            self.batch_processor = BatchProcessor(self.interface)
        else:
            self.batch_processor = None
    
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
        seed: Optional[int],
        preset: str = "none",
        enable_text_processing: bool = True,
        enable_audio_enhancement: bool = False,
        noise_reduction: float = 0.3,
        eq_preset: str = "vocal"
    ) -> Tuple[Optional[str], str]:
        """Generate speech with enhanced UI wrapper"""
        if not self.model_loaded:
            return None, "❌ Please load the model first."

        # Handle reference audio
        audio_path = None
        if reference_audio is not None:
            audio_path = reference_audio

        # Use preset if selected
        if preset and preset != "none" and ENHANCEMENTS_AVAILABLE:
            preset_exaggeration, preset_cfg_weight = emotion_presets.get_preset_parameters(preset)
            exaggeration = preset_exaggeration
            cfg_weight = preset_cfg_weight

        return self.interface.generate_speech(
            text=text,
            audio_prompt_path=audio_path,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            seed=seed,
            preset=preset if preset != "none" else None,
            enable_text_processing=enable_text_processing,
            enable_audio_enhancement=enable_audio_enhancement,
            noise_reduction=noise_reduction,
            eq_preset=eq_preset
        )
    
    def validate_reference_audio(self, audio_file) -> str:
        """Validate reference audio file"""
        if audio_file is None:
            return "ℹ️ No reference audio provided - will use default voice"

        is_valid, message = self.interface.validate_audio_file(audio_file)
        return message

    def get_preset_choices(self) -> List[str]:
        """Get available emotion/style presets"""
        if not ENHANCEMENTS_AVAILABLE:
            return ["none"]

        choices = ["none"]
        all_presets = emotion_presets.get_all_presets()
        for name, preset in all_presets.items():
            display_name = f"{preset.get('icon', '🎭')} {preset.get('name', name)}"
            choices.append((display_name, name))

        return choices

    def get_preset_info(self, preset_name: str) -> str:
        """Get information about selected preset"""
        if not ENHANCEMENTS_AVAILABLE or preset_name == "none":
            return "ℹ️ No preset selected"

        return emotion_presets.get_preset_info(preset_name)

    def process_batch_text(self, batch_text: str, **kwargs) -> str:
        """Process batch text input"""
        if not self.batch_processor:
            return "❌ Batch processing not available"

        if not batch_text.strip():
            return "❌ Please enter text for batch processing"

        # Split text by lines
        texts = [line.strip() for line in batch_text.split('\n') if line.strip()]

        if not texts:
            return "❌ No valid text lines found"

        item_ids = self.batch_processor.add_text_batch(texts, **kwargs)
        return f"✅ Added {len(item_ids)} items to batch queue"

    def process_batch_csv(self, csv_file, **kwargs) -> str:
        """Process batch CSV file"""
        if not self.batch_processor:
            return "❌ Batch processing not available"

        if csv_file is None:
            return "❌ Please upload a CSV file"

        try:
            with open(csv_file.name, 'r', encoding='utf-8') as f:
                csv_content = f.read()

            item_ids, message = self.batch_processor.add_csv_batch(csv_content, **kwargs)
            return message
        except Exception as e:
            return f"❌ Error reading CSV file: {str(e)}"

    def start_batch_processing(self) -> str:
        """Start batch processing"""
        if not self.batch_processor:
            return "❌ Batch processing not available"

        if not self.model_loaded:
            return "❌ Please load the model first"

        status = self.batch_processor.get_batch_status()
        if status['queue_size'] == 0:
            return "❌ No items in batch queue"

        if status['is_processing']:
            return "⚠️ Batch processing already in progress"

        self.batch_processor.process_batch()
        return f"🚀 Started processing {status['queue_size']} items"

    def get_batch_status(self) -> str:
        """Get batch processing status"""
        if not self.batch_processor:
            return "❌ Batch processing not available"

        status = self.batch_processor.get_batch_status()

        if status['is_processing']:
            progress = (status['processed_items'] / status['total_items'] * 100) if status['total_items'] > 0 else 0
            return f"""
            🔄 **Processing in progress...**

            📊 Progress: {status['processed_items']}/{status['total_items']} ({progress:.1f}%)
            ✅ Completed: {status['completed_items']}
            ❌ Failed: {status['failed_items']}

            🎵 Current: {status['current_item'] or 'None'}
            """
        else:
            return f"""
            ⏸️ **Processing stopped**

            📊 Total items: {status['total_items']}
            ✅ Completed: {status['completed_items']}
            ❌ Failed: {status['failed_items']}
            📋 Queue: {status['queue_size']}
            """

    def download_batch_results(self) -> Tuple[Optional[str], str]:
        """Create download for batch results"""
        if not self.batch_processor:
            return None, "❌ Batch processing not available"

        return self.batch_processor.create_batch_download()

    def clear_batch_queue(self) -> str:
        """Clear batch processing queue"""
        if not self.batch_processor:
            return "❌ Batch processing not available"

        self.batch_processor.clear_batch()
        return "✅ Batch queue cleared"
    
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
                <h1>🎤 ChatterBox TTS Enhanced</h1>
                <p>ResembleAI's State-of-the-Art Text-to-Speech Model</p>
                <p><em>Zero-shot TTS • Voice Cloning • Emotion Control • Audio Enhancement • Batch Processing</em></p>
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

            # Main interface with tabs
            with gr.Tabs():
                # Single Generation Tab
                with gr.TabItem("🎤 Single Generation"):
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

                            # Emotion/Style Presets
                            if ENHANCEMENTS_AVAILABLE:
                                preset_dropdown = gr.Dropdown(
                                    choices=self.get_preset_choices(),
                                    value="none",
                                    label="🎭 Emotion/Style Preset",
                                    interactive=True
                                )
                                preset_info = gr.Textbox(
                                    label="Preset Info",
                                    value="ℹ️ No preset selected",
                                    interactive=False,
                                    lines=3
                                )
                            else:
                                preset_dropdown = gr.Dropdown(
                                    choices=["none"],
                                    value="none",
                                    label="🎭 Emotion/Style Preset (Not Available)",
                                    interactive=False
                                )
                                preset_info = gr.Textbox(
                                    label="Preset Info",
                                    value="❌ Enhancement modules not loaded",
                                    interactive=False
                                )

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

                    # Audio Enhancement Section
                    gr.Markdown("## 🎚️ Audio Enhancement (Optional)")
                    with gr.Group(elem_classes=["control-panel"]):
                        enable_audio_enhancement = gr.Checkbox(
                            label="Enable Audio Enhancement",
                            value=False
                        )

                        with gr.Row():
                            noise_reduction = gr.Slider(
                                minimum=0.0,
                                maximum=1.0,
                                value=0.3,
                                step=0.1,
                                label="Noise Reduction",
                                visible=False
                            )

                            eq_preset = gr.Dropdown(
                                choices=["vocal", "balanced", "warm", "bright"],
                                value="vocal",
                                label="EQ Preset",
                                visible=False
                            )

                        enable_text_processing = gr.Checkbox(
                            label="Enable Text Processing (pronunciation, sound effects)",
                            value=True
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

                # Batch Processing Tab
                with gr.TabItem("📦 Batch Processing"):
                    if ENHANCEMENTS_AVAILABLE:
                        gr.Markdown("## 📝 Batch Input")

                        with gr.Tabs():
                            with gr.TabItem("Text Lines"):
                                batch_text_input = gr.Textbox(
                                    label="Batch Text (one line per generation)",
                                    placeholder="Enter multiple lines of text...\nEach line will be processed separately.",
                                    lines=8,
                                    max_lines=20
                                )

                            with gr.TabItem("CSV Upload"):
                                batch_csv_file = gr.File(
                                    label="Upload CSV File",
                                    file_types=[".csv"],
                                    file_count="single"
                                )
                                gr.Markdown("**CSV Format:** text,reference_audio,exaggeration,cfg_weight,seed,preset,filename")

                        # Batch controls
                        gr.Markdown("## ⚙️ Batch Settings")
                        with gr.Row():
                            with gr.Column():
                                batch_preset = gr.Dropdown(
                                    choices=self.get_preset_choices(),
                                    value="none",
                                    label="Default Preset for Batch"
                                )
                                batch_exaggeration = gr.Slider(
                                    minimum=0.0,
                                    maximum=1.0,
                                    value=0.5,
                                    step=0.1,
                                    label="Default Exaggeration"
                                )
                            with gr.Column():
                                batch_cfg_weight = gr.Slider(
                                    minimum=0.0,
                                    maximum=1.0,
                                    value=0.5,
                                    step=0.1,
                                    label="Default CFG Weight"
                                )
                                batch_reference_audio = gr.Audio(
                                    label="Default Reference Audio (optional)",
                                    type="filepath"
                                )

                        # Batch processing controls
                        gr.Markdown("## 🚀 Batch Processing")
                        with gr.Row():
                            add_batch_btn = gr.Button("📝 Add Text to Queue", variant="secondary")
                            add_csv_btn = gr.Button("📄 Add CSV to Queue", variant="secondary")
                            start_batch_btn = gr.Button("🚀 Start Processing", variant="primary")
                            clear_batch_btn = gr.Button("🗑️ Clear Queue", variant="stop")

                        # Batch status and results
                        batch_status = gr.Textbox(
                            label="Batch Status",
                            value="📋 Queue empty",
                            interactive=False,
                            lines=6
                        )

                        with gr.Row():
                            download_batch_btn = gr.Button("📥 Download Results", variant="secondary")
                            batch_download_file = gr.File(
                                label="Batch Results",
                                visible=False
                            )
                    else:
                        gr.Markdown("## ❌ Batch Processing Not Available")
                        gr.Markdown("Enhancement modules are not loaded. Please check your installation.")
            
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

            # Preset change handler
            if ENHANCEMENTS_AVAILABLE:
                preset_dropdown.change(
                    fn=self.get_preset_info,
                    inputs=[preset_dropdown],
                    outputs=[preset_info]
                )

            # Audio enhancement visibility
            enable_audio_enhancement.change(
                fn=lambda x: (gr.update(visible=x), gr.update(visible=x)),
                inputs=[enable_audio_enhancement],
                outputs=[noise_reduction, eq_preset]
            )

            generate_btn.click(
                fn=self.generate_speech_ui,
                inputs=[
                    text_input, reference_audio, exaggeration, cfg_weight, seed_input,
                    preset_dropdown, enable_text_processing, enable_audio_enhancement,
                    noise_reduction, eq_preset
                ],
                outputs=[output_audio, generation_status]
            )

            # Batch processing event handlers
            if ENHANCEMENTS_AVAILABLE:
                add_batch_btn.click(
                    fn=lambda text, preset, exag, cfg, ref_audio: self.process_batch_text(
                        text,
                        preset=preset if preset != "none" else None,
                        exaggeration=exag,
                        cfg_weight=cfg,
                        reference_audio=ref_audio
                    ),
                    inputs=[batch_text_input, batch_preset, batch_exaggeration, batch_cfg_weight, batch_reference_audio],
                    outputs=[batch_status]
                )

                add_csv_btn.click(
                    fn=lambda csv_file, preset, exag, cfg, ref_audio: self.process_batch_csv(
                        csv_file,
                        preset=preset if preset != "none" else None,
                        exaggeration=exag,
                        cfg_weight=cfg,
                        reference_audio=ref_audio
                    ),
                    inputs=[batch_csv_file, batch_preset, batch_exaggeration, batch_cfg_weight, batch_reference_audio],
                    outputs=[batch_status]
                )

                start_batch_btn.click(
                    fn=self.start_batch_processing,
                    outputs=[batch_status]
                )

                clear_batch_btn.click(
                    fn=self.clear_batch_queue,
                    outputs=[batch_status]
                )

                download_batch_btn.click(
                    fn=self.download_batch_results,
                    outputs=[batch_download_file, batch_status]
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
