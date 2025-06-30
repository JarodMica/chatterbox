import sys
from pathlib import Path

import gradio as gr
import pykakasi

# We now use direct file operations instead of gradio_utils

# Import ChatterboxTTS
from chatterbox.tts import ChatterboxTTS
from gradio_utils.utils import *
import json
import torch
import random
import logging

logger = logging.getLogger("gradio")
logger.setLevel(logging.INFO)

# Global variables for the Gradio app
tts_model = None
tts_path = None
kks = pykakasi.kakasi()
kks.setMode("H", "H")  # Hiragana to Hiragana
kks.setMode("K", "H")  # Katakana to Hiragana
kks.setMode("J", "H")  # Kanji to Hiragana
conv = kks.getConverter()


def normalize_japanese_text(text: str) -> str:
    """Normalize Japanese text to hiragana using pykakasi"""
    if not text:
        return text
    
    # Convert to hiragana
    hiragana_text = conv.do(text)
    return hiragana_text


def load_tts_model(t3_model_path, tokenizer_path=None, device="cpu"):
    """Load TTS model using from_specified method"""
    global tts_model
    
    with open("model_path.json", "r") as f:
        model_path = json.load(f)
        
    
    if not t3_model_path or not Path(t3_model_path).exists():
        raise gr.Error("Please select a valid T3 model file")
    
    # Default paths - these would need to be set based on your model structure
    voice_encoder_path = model_path["voice_encoder_path"]
    s3gen_path = model_path["s3gen_path"]
    
    # Use provided tokenizer path or default
    if not tokenizer_path or not Path(tokenizer_path).exists():
        tokenizer_path = model_path["tokenizer_path"]
    
    conds_path = model_path["conds_path"]
    
    try:
        tts_model = ChatterboxTTS.from_specified(
            voice_encoder_path=voice_encoder_path,
            t3_path=t3_model_path,
            s3gen_path=s3gen_path,
            tokenizer_path=tokenizer_path,
            conds_path=conds_path,
            device=device
        )
        return "Model loaded successfully!"
    except Exception as e:
        raise gr.Error(f"Failed to load model: {str(e)}")


def generate_speech(
    text: str,
    voice_file: str,
    t3_model_path: str,
    tokenizer_path: str,
    device: str,
    exaggeration: float,
    cfg_weight: float,
    temperature: float,
    normalize_japanese: bool,
    seed: int,
    redact: bool,
    translate_to: str,
    translation_strength: float,
    source_language: str,
    enable_language_converions: bool
):
    """Generate speech using the TTS model"""
    global tts_model, tts_path
    if not enable_language_converions:
        translation_strength = 0
    
    if seed ==-1:
        seed = random.randint(0, 1000000000)
        print(f"Random seed: {seed}")
    else:
        seed = int(seed)
        
    set_seed(seed)
    
    if not text.strip():
        raise gr.Error("Please enter text to generate")
    
    # Load model if not already loaded or if model path changed
    if tts_model is None or tts_path != t3_model_path:
        tts_path = t3_model_path
        load_tts_model(t3_model_path, tokenizer_path, device)
    
    # Normalize Japanese text if requested
    if normalize_japanese:
        text = normalize_japanese_text(text)
        
    if "default_voice.mp3" in voice_file:
        voice_file = None
    
    try:
        # Generate audio
        audio_tensor = tts_model.generate(
            text=text,
            audio_prompt_path=voice_file if voice_file else None,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            temperature=temperature,
            redact=redact,
            translate_to=translate_to,
            translation_strength=translation_strength,
            source_language=source_language
        )
        
        # Convert to numpy for Gradio
        audio_np = audio_tensor.squeeze().numpy()
        
        return (tts_model.sr, audio_np)
        
    except Exception as e:
        raise gr.Error(f"Generation failed: {str(e)}")

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def create_gradio_interface():
    """Create and return the Gradio interface"""
    
    # Create directories if they don't exist
    Path("voices").mkdir(exist_ok=True)
    Path("t3_models").mkdir(exist_ok=True)
    Path("tokenizers").mkdir(exist_ok=True)
    
    with gr.Blocks(title="Chatterbox TTS") as interface:
        gr.Markdown("# Chatterbox TTS Interface")
        gr.Markdown("Generate speech using the Chatterbox TTS model with customizable voice prompts and parameters.")
        
        with gr.Row():
            with gr.Column():
                t3_models = get_available_items("t3_models", valid_extensions=[".safetensors"], directory_only=False)
                tokenizer_files = get_available_items("tokenizers", valid_extensions=[".json"], directory_only=False)
                voice_files = get_available_items("voices", valid_extensions=[".wav", ".mp3", ".flac", ".m4a"], directory_only=False)
                
                # Model Selection
                gr.Markdown("### Model Configuration")
                t3_model_dropdown = gr.Dropdown(
                    choices=t3_models,
                    label="T3 Model",
                    info="Select a T3 model file from t3_models folder"
                )
                
                tokenizer_dropdown = gr.Dropdown(
                    choices=tokenizer_files,
                    label="Tokenizer",
                    info="Select a tokenizer file from tokenizers folder (optional)"
                )
                
                device_dropdown = gr.Dropdown(
                    choices=["cpu", "cuda", "mps"],
                    value="cpu",
                    label="Device",
                    info="Select computation device"
                )
                
                # Voice Selection
                gr.Markdown("### Voice Configuration")
                voice_dropdown = gr.Dropdown(
                    choices=voice_files,
                    value=voice_files[0] if len(voice_files) > 0 else "",
                    label="Voice File",
                    info="Select voice file from voices folder (optional - empty for default)"
                )
                enable_language_converions = gr.Checkbox(
                    label="Enable Language Conversions",
                    value=True,
                    info="Enable language conversions"
                )
                source_language_dropdown = gr.Dropdown(
                    choices=["japanese", "english",],
                    value="japanese",
                    label="Reference Audio Language",
                    info="Language of the reference audio"
                )
                translate_to_dropdown = gr.Dropdown(
                    choices=["japanese", "english",],
                    value="japanese",
                    label="Desired Language/Accent",
                    info="Language/accent to inference text with"
                )
                translation_strength_slider = gr.Slider(
                    minimum=0.0,
                    maximum=5.0,
                    value=1.0,
                    step=0.1,
                    label="Lang Conversion Strength",
                    info="Strength of conversion (reccomended 1-2)"
                )
                
                refresh_btn = gr.Button("Refresh Dropdowns")
                
            with gr.Column():
                # Text Input
                gr.Markdown("### Text Input")
                text_input = gr.Textbox(
                    label="Text to Generate",
                    placeholder="Enter text to convert to speech...",
                    lines=3
                )
                
                normalize_jp_checkbox = gr.Checkbox(
                    label="Normalize Japanese to Hiragana",
                    value=True,
                    info="Convert Japanese text to hiragana for better compatibility"
                )
                redact_checkbox = gr.Checkbox(
                    label="Redact Bracketed Text",
                    value=True,
                    info="Redact text within brackets"
                )
                
                
                # Inference Parameters
                gr.Markdown("### Inference Parameters")
                exaggeration_slider = gr.Slider(
                    minimum=0.0,
                    maximum=2.0,
                    value=0.5,
                    step=0.1,
                    label="Exaggeration",
                    info="Emotional intensity"
                )
                
                cfg_weight_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.5,
                    step=0.05,
                    label="CFG Weight",
                    info="Classifier-free guidance weight"
                )
                
                temperature_slider = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=0.8,
                    step=0.1,
                    label="Temperature",
                    info="Sampling temperature"
                )
                
                seed_input = gr.Number(
                    label="Seed",
                    value=0,
                    info="Random seed for reproducibility"
                )
        
        # Generation
        generate_btn = gr.Button("Generate Speech", variant="primary", size="lg")
        
        # Output
        audio_output = gr.Audio(
            label="Generated Audio",
            type="numpy"
        )
        
        voice_dropdown_root = gr.Textbox("voices", visible=False)
        voice_dropdown_valid_extensions = gr.Textbox("[.wav, .mp3, .flac, .m4a]", visible=False)
        voice_dropdown_directory_only = gr.Textbox("files", visible=False)
        
        t3_model_dropdown_root = gr.Textbox("t3_models", visible=False)
        t3_model_dropdown_valid_extensions = gr.Textbox("[.safetensors]", visible=False)
        t3_model_dropdown_directory_only = gr.Textbox("files", visible=False)
        
        tokenizer_dropdown_root = gr.Textbox("tokenizers", visible=False)
        tokenizer_dropdown_valid_extensions = gr.Textbox("[.json]", visible=False)
        tokenizer_dropdown_directory_only = gr.Textbox("files", visible=False)
        
        refresh_btn.click(
            fn=refresh_dropdown_proxy,
            inputs=[voice_dropdown_root, voice_dropdown_valid_extensions, voice_dropdown_directory_only, 
                     t3_model_dropdown_root, t3_model_dropdown_valid_extensions, t3_model_dropdown_directory_only, 
                     tokenizer_dropdown_root, tokenizer_dropdown_valid_extensions, tokenizer_dropdown_directory_only],
            outputs=[voice_dropdown, 
                     t3_model_dropdown, 
                     tokenizer_dropdown]
        )
        
        generate_btn.click(
            fn=generate_speech,
            inputs=[
                text_input,
                voice_dropdown,
                t3_model_dropdown,
                tokenizer_dropdown,
                device_dropdown,
                exaggeration_slider,
                cfg_weight_slider,
                temperature_slider,
                normalize_jp_checkbox,
                seed_input,
                redact_checkbox,
                translate_to_dropdown,
                translation_strength_slider,
                source_language_dropdown,
                enable_language_converions
            ],
            outputs=audio_output
        )
    
    return interface


def launch_gradio_app(share=False, server_port=7860):
    """Launch the Gradio application"""
    try:
        demo = create_gradio_interface()
        demo.launch(share=share, server_port=server_port)
    except Exception as e:
        print(f"Error launching Gradio app: {e}")
        print(f"Error type: {type(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    launch_gradio_app()