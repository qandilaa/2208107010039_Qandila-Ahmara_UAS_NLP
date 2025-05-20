import os
import uuid
import tempfile
import subprocess

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Path to TTS utilities
COQUI_DIR = os.path.join(BASE_DIR, "app", "coqui_utils")

# Path to TTS model
COQUI_MODEL_PATH = os.path.join(COQUI_DIR, "checkpoint_1260000-inference.pth")

# Path to configuration file
COQUI_CONFIG_PATH = os.path.join(COQUI_DIR, "config.json")

# Speaker name
COQUI_SPEAKER = "wibowo"

def transcribe_text_to_speech(text: str) -> str:
    """
    Convert text to speech using the specified TTS engine
    
    Args:
        text (str): Text to convert to speech
        
    Returns:
        str: Path to the generated audio file
    """
    # Check if files exist
    if not os.path.exists(COQUI_MODEL_PATH):
        return f"[ERROR] TTS model not found at {COQUI_MODEL_PATH}"
        
    if not os.path.exists(COQUI_CONFIG_PATH):
        return f"[ERROR] TTS config not found at {COQUI_CONFIG_PATH}"
    
    # Use Coqui TTS
    path = _tts_with_coqui(text)
    return path

def _tts_with_coqui(text: str) -> str:
    """
    Use Coqui TTS to synthesize speech
    
    Args:
        text (str): Text to convert to speech
        
    Returns:
        str: Path to the generated audio file
    """
    tmp_dir = tempfile.gettempdir()
    output_path = os.path.join(tmp_dir, f"tts_{uuid.uuid4()}.wav")

    # Run Coqui TTS with subprocess
    cmd = [
        "tts",
        "--text", text,
        "--model_path", COQUI_MODEL_PATH,
        "--config_path", COQUI_CONFIG_PATH,
        "--speaker_idx", COQUI_SPEAKER,
        "--out_path", output_path
    ]
    
    try:
        process = subprocess.run(cmd, check=True, capture_output=True)
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return output_path
        else:
            return f"[ERROR] TTS output file is empty or not created: {process.stderr.decode() if hasattr(process, 'stderr') else 'No error info'}"
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] TTS subprocess failed: {e}")
        error_details = e.stderr.decode() if hasattr(e, 'stderr') else 'No error details'
        return f"[ERROR] Failed to synthesize speech: {error_details}"