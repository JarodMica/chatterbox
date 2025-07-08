try:
    from beam import function, Image, Volume, endpoint
    IMAGE = Image(
    python_version="python3.11",
    python_packages=[
        "https://github.com/JarodMica/chatterbox.git"
    ]
    ).add_python_packages(
        [
            "huggingface_hub",
            "huggingface_hub[hf-transfer]",
        ]
    ).with_envs(
        "HF_HUB_ENABLE_HF_TRANSFER=1"
    )
except:
    print("Beam not installed, skipping import")

import sys
from pathlib import Path

import gradio as gr
import pykakasi
import requests
import io
import base64
import json
import numpy as np
import soundfile as sf
import yaml
import threading
import time
import queue
import os

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
beam_endpoint_url = None
beam_auth_token = None
beam_deployment_id = None
log_stream_thread = None
log_stream_active = False
log_queue = queue.Queue()
log_history = []  # Store log history for scrolling window
kks = pykakasi.kakasi()
kks.setMode("H", "H")  # Hiragana to Hiragana
kks.setMode("K", "H")  # Katakana to Hiragana
kks.setMode("J", "H")  # Kanji to Hiragana
conv = kks.getConverter()

# Beam deployment parameters
with open("beam_config.yaml", "r") as f:
    config = yaml.safe_load(f)
BEAM_MEMORY = str(config["Memory"])
BEAM_CPU = int(config["CPU"])
BEAM_GPU = str(config["GPU"])
CHATTERBOX_PROJECT = "./chatterbox-project"
T3_VOLUME = "./t3_models"


def get_subprocess_env():
    import os
    env = os.environ.copy()
    env['LANG'] = 'en_US.UTF-8'
    env['LC_ALL'] = 'en_US.UTF-8'
    return env


def normalize_japanese_text(text: str) -> str:
    if not text:
        return text
    
    hiragana_text = conv.do(text)
    return hiragana_text


def load_tts_model(t3_model_path, tokenizer_path=None, device="cpu"):
    global tts_model
    
    with open("model_path.json", "r") as f:
        model_path = json.load(f)
        
    
    if not t3_model_path or not Path(t3_model_path).exists():
        raise gr.Error("Please select a valid T3 model file")
    
    voice_encoder_path = model_path["voice_encoder_path"]
    s3gen_path = model_path["s3gen_path"]
    
    if not tokenizer_path or not Path(tokenizer_path).exists():
        tokenizer_path = model_path["tokenizer_path"]
    
    conds_path = Path(model_path["conds_path"])
    
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



try:
    @endpoint(
        name="chatterbox-inference",
        image=IMAGE,
        memory=BEAM_MEMORY,
        cpu=BEAM_CPU,
        gpu=BEAM_GPU,
        volumes=[Volume(name="chatterbox-project", mount_path=CHATTERBOX_PROJECT), Volume(name="t3_models", mount_path=T3_VOLUME)],
        keep_warm_seconds=120
        # timeout=-1
    )
    def generate_speech_beam(
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
            
            audio_np = audio_tensor.squeeze().numpy()
            
            # Return JSON-serializable format for Beam endpoint
            return {
                "sample_rate": int(tts_model.sr),
                "audio_data": audio_np.tolist()
            }
            
        except Exception as e:
            raise gr.Error(f"Generation failed: {str(e)}")
except:
    print("Beam not installed, skipping import")
    

def generate_speech_local(
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
    
    if tts_model is None or tts_path != t3_model_path:
        tts_path = t3_model_path
        load_tts_model(t3_model_path, tokenizer_path, device)
    
    if normalize_japanese:
        text = normalize_japanese_text(text)
        
    if "default_voice.mp3" in voice_file:
        voice_file = None
    
    try:
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
        
        # Convert to numpy and return as tuple with sample rate for Gradio
        return (int(tts_model.sr), audio_tensor.squeeze().numpy())
    except Exception as e:
        raise gr.Error(f"Generation failed: {str(e)}")
    
def generate_proxy(
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
    enable_language_converions: bool,
    use_beam: bool,
    use_default_models: bool
):
    default_used = False
    if use_default_models:
        default_used = True
        t3_model_path = os.path.join(CHATTERBOX_PROJECT, "chatterbox_weights/t3_cfg.safetensors")
        tokenizer_path = os.path.join(CHATTERBOX_PROJECT, "chatterbox_weights/tokenizer.json")
        
    if use_beam:
        assert t3_model_path, "t3_model_path is required for beam inference"
        
        if not check_t3_model_exists_in_beam(t3_model_path) and not default_used:
            logger.info(f"Validating model exists in beam for: {t3_model_path}")
            raise gr.Error('Model needs to be uploaded first! Click "Get Upload Command" and copy and paste the command into your terminal')
        
        # Convert Windows paths to Unix paths for Linux server
        voice_file_unix = voice_file.replace('\\', '/') if voice_file else voice_file
        t3_model_path_unix = t3_model_path.replace('\\', '/') if t3_model_path else t3_model_path
        tokenizer_path_unix = tokenizer_path.replace('\\', '/') if tokenizer_path else tokenizer_path
        
        return call_beam_endpoint(
            text,
            voice_file_unix,
            t3_model_path_unix,
            tokenizer_path_unix,
            device,
            exaggeration,
            cfg_weight,
            temperature,
            normalize_japanese,
            seed,
            redact,
            translate_to,
            translation_strength,
            source_language,
            enable_language_converions
        )
    else:
        return generate_speech_local(
            text,
            voice_file,
            t3_model_path,
            tokenizer_path,
            device,
            exaggeration,
            cfg_weight,
            temperature,
            normalize_japanese,
            seed,
            redact,
            translate_to,
            translation_strength,
            source_language,
            enable_language_converions
        )

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def launch_new_beam_deployment():
    try:
        import subprocess
        result = subprocess.run(['uv', 'run', 'beam', 'deploy', 'gradio_interface.py:generate_speech_beam'], 
                              capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            endpoint_url = None
            bearer_token = None
            deployment_id = None
            
            for line in lines:
                if 'https://' in line and 'app.beam.cloud' in line:
                    if 'curl' in line:
                        url_start = line.find('https://')
                        url_end = line.find("'", url_start)
                        if url_end == -1:
                            url_end = line.find(' ', url_start)
                        if url_end == -1:
                            url_end = len(line)
                        endpoint_url = line[url_start:url_end]
                
                if 'Authorization: Bearer' in line:
                    token_start = line.find('Bearer ') + 7
                    token_end = line.find("'", token_start)
                    if token_end == -1:
                        token_end = line.find(' ', token_start)
                    if token_end == -1:
                        token_end = len(line)
                    bearer_token = line[token_start:token_end]
            
            # Always get the actual deployment ID from beam deployment list after deployment
            try:
                import subprocess
                list_result = subprocess.run(['uv', 'run', 'beam', 'deployment', 'list', '--format', 'json'], 
                                           capture_output=True, text=True, timeout=10)
                if list_result.returncode == 0:
                    deployments = json.loads(list_result.stdout)
                    for deployment in deployments:
                        if deployment.get('name') == 'chatterbox-inference' and deployment.get('active'):
                            deployment_id = deployment.get('id')
                            logger.info(f"Found deployment ID from list: {deployment_id}")
                            break
            except Exception as e:
                logger.error(f"Failed to get deployment ID from list: {str(e)}")
            
            global beam_endpoint_url, beam_auth_token, beam_deployment_id
            if endpoint_url:
                beam_endpoint_url = endpoint_url
            if bearer_token:
                beam_auth_token = bearer_token
            if deployment_id:
                beam_deployment_id = deployment_id
            
            status_msg = f"✅ Beam deployment successful!\nEndpoint: {endpoint_url}"
            if bearer_token:
                status_msg += f"\nAuth token captured: {bearer_token[:20]}..."
            if deployment_id:
                status_msg += f"\nDeployment ID: {deployment_id}"
            
            save_beam_deployment_info()
            status_msg += f"\nDeployment info saved to beam_deployment.yaml"
            status_msg += f"\n\nFull output:\n{result.stdout}"
            
            return status_msg
        else:
            return f"❌ Beam deployment failed:\n{result.stderr}\n\nOutput:\n{result.stdout}"
    
    except subprocess.TimeoutExpired:
        return "❌ Beam deployment timed out after 5 minutes"
    except Exception as e:
        return f"❌ Beam deployment error: {str(e)}"


def call_beam_endpoint(
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
) -> tuple:
    global beam_endpoint_url, beam_auth_token
    
    assert t3_model_path, "t3_model_path is required for beam endpoint"
    
    if not beam_endpoint_url:
        raise gr.Error("Beam endpoint not deployed. Please launch beam deployment first.")
    
    # Get auth token from environment or prompt user
    if not beam_auth_token:
        beam_auth_token = os.getenv('BEAM_AUTH_TOKEN')
        if not beam_auth_token:
            raise gr.Error("BEAM_AUTH_TOKEN environment variable not set. Please set your Beam auth token.")
    
    headers = {
        "Connection": "keep-alive",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {beam_auth_token}",
    }
    
    data = {
        "text": text,
        "voice_file": voice_file,
        "t3_model_path": t3_model_path,
        "tokenizer_path": tokenizer_path,
        "device": device,
        "exaggeration": exaggeration,
        "cfg_weight": cfg_weight,
        "temperature": temperature,
        "normalize_japanese": normalize_japanese,
        "seed": seed,
        "redact": redact,
        "translate_to": translate_to,
        "translation_strength": translation_strength,
        "source_language": source_language,
        "enable_language_converions": enable_language_converions
    }
    
    try:
        logger.info(f"Calling Beam endpoint: {beam_endpoint_url}")
        response = requests.post(beam_endpoint_url, headers=headers, json=data, timeout=180)
        
        if response.status_code == 200:
            result = response.json()
            
            # Handle different response formats
            if isinstance(result, dict):
                if 'audio_data' in result and 'sample_rate' in result:
                    # Direct audio data response (from our modified generate_speech)
                    audio_data = np.array(result['audio_data'])
                    sample_rate = result['sample_rate']
                    return (sample_rate, audio_data)
                elif 'audio_base64' in result:
                    # Base64 encoded audio
                    audio_bytes = base64.b64decode(result['audio_base64'])
                    audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes))
                    return (sample_rate, audio_data)
                else:
                    raise gr.Error(f"Unexpected response format from Beam endpoint: {result}")
            
            # If we get here, try to interpret as direct audio data (legacy)
            if isinstance(result, (list, tuple)) and len(result) == 2:
                return tuple(result)
            else:
                raise gr.Error(f"Unexpected response format from Beam endpoint: {type(result)}")
                
        elif response.status_code == 503:
            raise gr.Error("Beam server is not available. The container may have spun down. Please relaunch the deployment.")
        elif response.status_code == 401:
            raise gr.Error("Authentication failed. Please check your BEAM_AUTH_TOKEN.")
        else:
            raise gr.Error(f"Beam endpoint error: {response.status_code} - {response.text}")
            
    except requests.exceptions.Timeout:
        raise gr.Error("Beam endpoint request timed out. The container may be cold-starting.")
    except requests.exceptions.ConnectionError:
        raise gr.Error("Cannot connect to Beam endpoint. Please check if the deployment is active.")


def save_beam_deployment_info():
    global beam_endpoint_url, beam_auth_token, beam_deployment_id
    
    from datetime import datetime
    
    deployment_info = {
        'endpoint_url': beam_endpoint_url,
        'auth_token': beam_auth_token,
        'deployment_id': beam_deployment_id,
        'timestamp': str(datetime.now())
    }
    
    try:
        with open('beam_deployment.yaml', 'w') as f:
            yaml.dump(deployment_info, f, default_flow_style=False)
        logger.info("Saved beam deployment info to beam_deployment.yaml")
    except Exception as e:
        logger.error(f"Failed to save deployment info: {str(e)}")


def load_beam_deployment_info():
    global beam_endpoint_url, beam_auth_token, beam_deployment_id
    
    try:
        if Path('beam_deployment.yaml').exists():
            with open('beam_deployment.yaml', 'r') as f:
                deployment_info = yaml.safe_load(f)
            
            beam_endpoint_url = deployment_info.get('endpoint_url')
            beam_auth_token = deployment_info.get('auth_token')
            beam_deployment_id = deployment_info.get('deployment_id')
            
            logger.info("Loaded beam deployment info from beam_deployment.yaml")
            return True
    except Exception as e:
        logger.error(f"Failed to load deployment info: {str(e)}")
    
    return False


def check_existing_deployment():
    try:
        import subprocess
        env = get_subprocess_env()
        result = subprocess.run(['uv', 'run', 'beam', 'deployment', 'list', '--format', 'json'], 
                              capture_output=True, text=True, timeout=30, env=env, encoding='utf-8')
        
        if result.returncode == 0:
            import json
            deployments = json.loads(result.stdout)
            
            # Look for chatterbox-inference deployments (active or inactive)
            for deployment in deployments:
                if deployment.get('name') == 'chatterbox-inference':
                    deployment_id = deployment.get('id')
                    name = deployment.get('name')
                    is_active = deployment.get('active', False)
                    logger.info(f"Found deployment: {deployment_id} ({name}) - Active: {is_active}")
                    return deployment_id, is_active
            
            return None, False
        else:
            logger.error(f"Failed to check deployments: {result.stderr}")
            return None, False
            
    except Exception as e:
        logger.error(f"Error checking existing deployment: {str(e)}")
        return None, False


def start_deployment(deployment_id):
    try:
        import subprocess
        result = subprocess.run(['uv', 'run', 'beam', 'deployment', 'start', deployment_id], 
                              capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            logger.info(f"Successfully started deployment {deployment_id}")
            return True, result.stdout
        else:
            logger.error(f"Failed to start deployment {deployment_id}: {result.stderr}")
            return False, result.stderr
            
    except Exception as e:
        logger.error(f"Error starting deployment {deployment_id}: {str(e)}")
        return False, str(e)


def smart_beam_deployment():
    global beam_endpoint_url, beam_auth_token, beam_deployment_id
    
    try:
        import subprocess
        
        # Always check for existing deployments first
        env = get_subprocess_env()
        result = subprocess.run(['uv', 'run', 'beam', 'deployment', 'list', '--format', 'json'], 
                              capture_output=True, text=True, timeout=30, env=env, encoding='utf-8')
        
        existing_deployments = []
        if result.returncode == 0:
            import json
            deployments = json.loads(result.stdout)
            
            # Find all chatterbox-inference deployments
            for deployment in deployments:
                if deployment.get('name') == 'chatterbox-inference':
                    deployment_id = deployment.get('id')
                    is_active = deployment.get('active', False)
                    if deployment_id:
                        existing_deployments.append((deployment_id, is_active))
                        logger.info(f"Found existing deployment: {deployment_id} - Active: {is_active}")
        
        if len(existing_deployments) > 1:
            logger.info(f"Found {len(existing_deployments)} existing deployments, cleaning up...")
            result_msg = f"🧹 Found {len(existing_deployments)} existing deployments, cleaning up...\n\n"
            
            for deployment_id, is_active in existing_deployments:
                result_msg += f"Processing deployment: {deployment_id}\n"
                
                if is_active:
                    logger.info(f"Stopping deployment {deployment_id}")
                    stop_result = subprocess.run(['uv', 'run', 'beam', 'deployment', 'stop', deployment_id], 
                                               capture_output=True, text=True, timeout=60, env=env, encoding='utf-8')
                    if stop_result.returncode == 0:
                        result_msg += f"✅ Stopped {deployment_id}\n"
                    else:
                        result_msg += f"⚠️ Failed to stop {deployment_id}: {stop_result.stderr}\n"
                
                logger.info(f"Deleting deployment {deployment_id}")
                delete_result = subprocess.run(['uv', 'run', 'beam', 'deployment', 'delete', deployment_id], 
                                             capture_output=True, text=True, timeout=60, env=env, encoding='utf-8')
                if delete_result.returncode == 0:
                    result_msg += f"✅ Deleted {deployment_id}\n"
                else:
                    result_msg += f"⚠️ Failed to delete {deployment_id}: {delete_result.stderr}\n"
            
            result_msg += "\n🚀 Creating fresh deployment...\n"
            new_deployment_result = launch_new_beam_deployment()
            return result_msg + new_deployment_result
        
        elif len(existing_deployments) == 1:
            deployment_id, is_active = existing_deployments[0]
            beam_deployment_id = deployment_id
            
            if is_active:
                if load_beam_deployment_info() and beam_deployment_id == deployment_id:
                    return f"✅ Using existing deployment!\nEndpoint: {beam_endpoint_url}\nDeployment ID: {beam_deployment_id}"
                else:
                    return f"✅ Found active deployment!\nDeployment ID: {deployment_id}\n⚠️ You may need to redeploy to capture endpoint URL and auth token for full functionality."
            else:
                success, output = start_deployment(deployment_id)
                if success:
                    return f"✅ Restarted existing deployment!\nDeployment ID: {deployment_id}\nRestart output:\n{output}"
                else:
                    return f"❌ Failed to restart deployment {deployment_id}:\n{output}\n\nCreating new deployment...\n\n" + launch_new_beam_deployment()
        
        else:
            logger.info("No existing deployments found, creating new one")
            return launch_new_beam_deployment()
            
    except Exception as e:
        logger.error(f"Error in smart deployment: {str(e)}")
        return f"❌ Error checking deployments: {str(e)}\n\nTrying to create new deployment...\n\n" + launch_new_beam_deployment()


def recreate_beam_deployment():
    global beam_endpoint_url, beam_auth_token, beam_deployment_id
    
    try:
        import subprocess
        
        logger.info("Starting deployment recreation process")
        result_msg = "🔄 Recreating Beam Deployment...\n\n"
        
        existing_id, is_active = check_existing_deployment()
        if existing_id:
            result_msg += f"Found existing deployment: {existing_id}\n"
            
            if is_active:
                logger.info(f"Stopping active deployment {existing_id}")
                result_msg += "⏹️ Stopping deployment...\n"
                stop_result = subprocess.run(['uv', 'run', 'beam', 'deployment', 'stop', existing_id], 
                                           capture_output=True, text=True, timeout=60)
                if stop_result.returncode == 0:
                    result_msg += "✅ Deployment stopped successfully\n"
                    logger.info(f"Successfully stopped deployment {existing_id}")
                else:
                    result_msg += f"⚠️ Failed to stop deployment: {stop_result.stderr}\n"
                    logger.warning(f"Failed to stop deployment {existing_id}: {stop_result.stderr}")
            
            logger.info(f"Deleting deployment {existing_id}")
            result_msg += "🗑️ Deleting deployment...\n"
            delete_result = subprocess.run(['uv', 'run', 'beam', 'deployment', 'delete', existing_id], 
                                         capture_output=True, text=True, timeout=60)
            if delete_result.returncode == 0:
                result_msg += "✅ Deployment deleted successfully\n"
                logger.info(f"Successfully deleted deployment {existing_id}")
            else:
                result_msg += f"⚠️ Failed to delete deployment: {delete_result.stderr}\n"
                logger.warning(f"Failed to delete deployment {existing_id}: {delete_result.stderr}")
        else:
            result_msg += "No existing deployment found\n"
            logger.info("No existing deployment found to recreate")
        
        beam_endpoint_url = None
        beam_auth_token = None
        beam_deployment_id = None
        
        result_msg += "\n🚀 Creating new deployment...\n"
        logger.info("Creating new deployment")
        new_deployment_result = launch_new_beam_deployment()
        
        return result_msg + new_deployment_result
        
    except Exception as e:
        logger.error(f"Error recreating deployment: {str(e)}")
        return f"❌ Error recreating deployment: {str(e)}"


def get_deployment_id():
    try:
        import subprocess
        result = subprocess.run(['uv', 'run', 'beam', 'deployment', 'list', '--format', 'json'], 
                              capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            import json
            deployments = json.loads(result.stdout)
            
            # Find the most recent active chatterbox-inference deployment
            for deployment in deployments:
                if deployment.get('name') == 'chatterbox-inference' and deployment.get('active'):
                    deployment_id = deployment.get('id')
                    logger.info(f"Found deployment ID: {deployment_id}")
                    return deployment_id
            
            return None
        else:
            logger.error(f"Failed to get deployment list: {result.stderr}")
            return None
            
    except Exception as e:
        logger.error(f"Error getting deployment ID: {str(e)}")
        return None


def log_stream_worker():
    global log_stream_active, beam_deployment_id, log_queue
    
    if not beam_deployment_id:
        log_queue.put("❌ No deployment ID available.")
        return
    
    try:
        import subprocess
        
        process = subprocess.Popen(
            ['uv', 'run', 'beam', 'logs', '--deployment-id', beam_deployment_id],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        log_queue.put(f"🔄 Starting log stream for deployment {beam_deployment_id}...\n")
        
        while log_stream_active and process.poll() is None:
            line = process.stdout.readline()
            if line:
                log_queue.put(line.rstrip())
            else:
                time.sleep(0.1)  # Small delay to prevent busy waiting
        
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        
        if log_stream_active:
            log_queue.put("\n🔄 Log stream ended.")
        else:
            log_queue.put("\n⏹️ Log stream stopped by user.")
            
    except Exception as e:
        log_queue.put(f"\n❌ Error in log stream: {str(e)}")


def start_log_stream():
    global log_stream_thread, log_stream_active
    
    if not beam_deployment_id:
        return "❌ No deployment ID available. Please launch deployment first."
    
    if log_stream_active:
        return "🔄 Log stream is already running. Click 'Stop Log Stream' to stop it first."
    
    # Start the streaming thread
    log_stream_active = True
    log_stream_thread = threading.Thread(target=log_stream_worker, daemon=True)
    log_stream_thread.start()
    
    return f"🔄 Started log streaming for deployment {beam_deployment_id}..."


def stop_log_stream():
    global log_stream_active
    
    if not log_stream_active:
        return "⏹️ Log stream is not currently running."
    
    log_stream_active = False
    return "⏹️ Stopping log stream..."


def get_streaming_logs():
    global log_queue, log_history
    
    new_logs = []
    
    while not log_queue.empty():
        try:
            line = log_queue.get_nowait()
            new_logs.append(line)
        except queue.Empty:
            break
    
    max_history = 1000
    if new_logs:
        log_history.extend(new_logs)
        
        # Trim history to maintain 1000-line window
        if len(log_history) > max_history:
            log_history = log_history[-max_history:]
    
    if log_history:
        if len(log_history) == max_history:
            result = f"... (showing last {max_history} lines)\n" + "\n".join(log_history)
        else:
            result = "\n".join(log_history)
        return result
    
    return ""


def clear_logs_display():
    global log_queue, log_history
    
    while not log_queue.empty():
        try:
            log_queue.get_nowait()
        except queue.Empty:
            break
    
    log_history.clear()
    
    return ""


def check_t3_model_exists_in_beam(t3_model_path: str) -> bool:
    assert t3_model_path, "t3_model_path cannot be empty"
    
    try:
        import subprocess
        
        model_filename = Path(t3_model_path).name
        logger.info(f"Checking if model {model_filename} exists in beam t3_models")
        
        env = get_subprocess_env()
        subprocess.run(['uv', 'run', 'beam', 'volume', 'create', 't3_models'], 
                       capture_output=True, text=True, timeout=30, env=env, encoding='utf-8')
        
        result = subprocess.run(['uv', 'run', 'beam', 'ls', 't3_models'], 
                              capture_output=True, text=True, timeout=30, env=env, encoding='utf-8')
        
        if result.returncode == 0:
            model_exists = model_filename in result.stdout
            logger.info(f"Model {model_filename} exists in beam: {model_exists}")
            return model_exists
        else:
            logger.error(f"Failed to list beam t3_models: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"Error checking t3 model in beam: {str(e)}")
        return False


def upload_t3_model_to_beam(t3_model_path: str) -> str:
    """Provide upload command for user to copy and paste"""
    if not t3_model_path:
        raise gr.Error("Select a t3_model first, t3_model_path cannot be empty")
    
    try:
        model_path = Path(t3_model_path)
        if not model_path.exists():
            return f"❌ Model file not found: {t3_model_path}"
        
        model_filename = model_path.name
        logger.info(f"Checking if {model_filename} already exists in beam")
        
        if check_t3_model_exists_in_beam(t3_model_path):
            logger.info(f"Model {model_filename} already exists in beam, no upload needed")
            return f"ℹ️ Model {model_filename} already exists in beam t3_models, no upload needed"
        
        # Create the upload command
        upload_command = f"uv run beam cp {t3_model_path} beam://t3_models"
        
        # Print to terminal for easy copy-paste
        print("\n" + "="*60)
        print("📋 COPY AND PASTE THIS COMMAND IN YOUR TERMINAL:")
        print("="*60)
        print(upload_command)
        print("="*60)
        print(f"This will upload {model_filename} to beam t3_models")
        print("="*60 + "\n")
        
        logger.info(f"Upload command for {model_filename}: {upload_command}")
        
        return upload_command

        
    except Exception as e:
        return gr.Error(f"Error: {str(e)}")



def create_gradio_interface():
    
    Path("voices").mkdir(exist_ok=True)
    Path("t3_models").mkdir(exist_ok=True)
    Path("tokenizers").mkdir(exist_ok=True)
    
    load_beam_deployment_info()
    
    with gr.Blocks(title="Chatterbox TTS") as interface:
        gr.Markdown("# Chatterbox TTS Interface")
        gr.Markdown("Generate speech using the Chatterbox TTS model with customizable voice prompts and parameters.")
        
        with gr.Row():
            # Left Column - Model Configuration and Beam Controls
            with gr.Column(scale=2):
                
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
                use_default_models_checkbox = gr.Checkbox(
                    label="Use Default Models",
                    value=False,
                    info="Turn on to use base Chatterbox Model"
                )
                
                tokenizer_dropdown = gr.Dropdown(
                    choices=tokenizer_files,
                    label="Tokenizer",
                    info="Select a tokenizer file from tokenizers folder (optional)"
                )
                
                device_dropdown = gr.Dropdown(
                    choices=["cpu", "cuda", "mps"],
                    value="cuda",
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
                with gr.Accordion("Language Conversion", open=False):
                    enable_language_converions = gr.Checkbox(
                        label="Enable Language Conversions",
                        value=False,
                        info="Enable language conversions"
                    )
                    source_language_dropdown = gr.Dropdown(
                        choices=["japanese", "english"],
                        value="english",
                        label="Reference Audio Language",
                        info="Language of the reference audio"
                    )
                    translate_to_dropdown = gr.Dropdown(
                        choices=["japanese", "english"],
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
                
            # Middle Column - Text Input and Inference Parameters
            with gr.Column(scale=2):
                # Text Input
                gr.Markdown("### Text Controls")
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
            
                
            with gr.Column(scale=2):
                gr.Markdown("### Beam Deployment Controls")
                with gr.Accordion("Beam", open=False):
                    use_beam_checkbox = gr.Checkbox(
                        label="Use Beam",
                        value=False,
                        info="Use Beam for faster generation"
                    )
                    
                    gr.Markdown("#### Model Upload")
                    upload_model_btn = gr.Button("Get Upload Command", variant="secondary")
                    upload_status_textbox = gr.Textbox(
                        label="COPY AND PASTE THIS COMMAND IN YOUR TERMINAL:",
                        placeholder="Click 'Get Upload Command' to generate the beam cp command...",
                        lines=1,
                        max_lines=2,
                        interactive=False,
                        show_copy_button=True
                    )
                    
                    gr.Markdown("#### Deployment Management")
                    with gr.Row():
                        launch_beam_btn = gr.Button("Launch Beam Deployment", variant="secondary")
                        recreate_beam_btn = gr.Button("Recreate Deployment", variant="secondary")
                    
                    beam_status_textbox = gr.Textbox(
                        label="Beam Status",
                        placeholder="Click 'Launch Beam Deployment' to deploy your endpoint...",
                        lines=6,
                        max_lines=6,
                        interactive=False
                    )
                    
                    gr.Markdown("#### Container Logs")
                    with gr.Row():
                        start_logs_btn = gr.Button("Start Log Stream", variant="primary")
                        stop_logs_btn = gr.Button("Stop Log Stream", variant="secondary")
                        clear_logs_btn = gr.Button("Clear Logs", variant="secondary")
                    
                    logs_output = gr.Textbox(
                        label="Container Logs (Live Stream)",
                        placeholder="Click 'Start Log Stream' to view real-time container logs...",
                        lines=10,
                        max_lines=10,
                        interactive=False,
                        show_copy_button=True
                    )
                
        with gr.Row():
            text_input = gr.Textbox(
                label="Text to Generate",
                placeholder="Enter text to convert to speech...",
                lines=3
            )
        with gr.Row():
            generate_btn = gr.Button("Generate Speech", variant="primary", size="lg")
            
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
        
        launch_beam_btn.click(
            fn=smart_beam_deployment,
            outputs=beam_status_textbox
        )
        
        recreate_beam_btn.click(
            fn=recreate_beam_deployment,
            outputs=beam_status_textbox
        )
        
        start_logs_btn.click(
            fn=start_log_stream,
            outputs=logs_output
        )
        
        stop_logs_btn.click(
            fn=stop_log_stream,
            outputs=logs_output
        )
        
        clear_logs_btn.click(
            fn=clear_logs_display,
            outputs=logs_output
        )
        
        upload_model_btn.click(
            fn=upload_t3_model_to_beam,
            inputs=t3_model_dropdown,
            outputs=upload_status_textbox
        )
        
        generate_btn.click(
            fn=generate_proxy,
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
                enable_language_converions,
                use_beam_checkbox,
                use_default_models_checkbox
            ],
            outputs=audio_output
        )
        
        # Set up automatic log updates every 2 seconds when streaming is active
        def update_logs_if_streaming():
            if log_stream_active:
                new_logs = get_streaming_logs()
                if new_logs:
                    return new_logs
            return gr.update()  # No update if not streaming or no new logs
        
        # Create a timer that updates logs every 2 seconds
        interface.load(
            fn=update_logs_if_streaming,
            outputs=logs_output,
            every=2
        )
    
    return interface


def launch_gradio_app(share=False, server_port=7860):
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