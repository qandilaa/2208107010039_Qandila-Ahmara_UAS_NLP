import os
import tempfile
import requests
import gradio as gr
import scipy.io.wavfile
import numpy as np
import json

def voice_chat(audio, chat_history=None):
    if audio is None:
        return None, chat_history or []
    
    # Initialize chat history if None
    if chat_history is None:
        chat_history = []
    
    sr, audio_data = audio
    
    # Save input audio to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        scipy.io.wavfile.write(tmpfile.name, sr, audio_data)
        audio_path = tmpfile.name
    
    try:
        # Send file to FastAPI backend
        with open(audio_path, "rb") as f:
            files = {"file": ("voice.wav", f, "audio/wav")}
            response = requests.post("http://localhost:8000/voice-chat", files=files)
        
        # Clean up temporary file
        os.unlink(audio_path)
        
        if response.status_code == 200:
            # Get JSON response that includes transcription, response text and audio
            data = response.json()
            
            # Get the audio file URL
            audio_url = data.get("audio_url")
            
            # Add transcription and response to chat history
            user_message = data.get("transcription", "")
            bot_message = data.get("response", "")
            
            chat_history.append((user_message, bot_message))
            
            # Get the audio from the URL
            audio_response = requests.get(f"http://localhost:8000{audio_url}")
            
            # Save to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as out_file:
                out_file.write(audio_response.content)
                out_file_path = out_file.name
            
            # Read audio file and return
            sr, audio_data = scipy.io.wavfile.read(out_file_path)
            # Clean up
            os.unlink(out_file_path)
            
            return (sr, audio_data.astype(np.float32) / 32768.0), chat_history
        else:
            error_msg = f"Error: {response.status_code} - {response.text}"
            chat_history.append(("Error occurred", error_msg))
            return None, chat_history
            
    except Exception as e:
        error_msg = f"Connection error: {str(e)}"
        chat_history.append(("Error occurred", error_msg))
        return None, chat_history

# Create Gradio UI with improved design
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎙️ Voice Chatbot")
    gr.Markdown("Speak into the microphone and get voice responses from the AI assistant.")
    
    chatbot = gr.Chatbot(height=400, label="Conversation")
    
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(
                sources="microphone", 
                type="numpy", 
                label="🎤 Record Your Question",
                interactive=True
            )
            
        with gr.Column():
            audio_output = gr.Audio(
                type="numpy", 
                label="🔊 Assistant Response",
                autoplay=True,
                interactive=False
            )
    
    submit_btn = gr.Button("🔁 Submit", variant="primary")
    clear_btn = gr.Button("🗑️ Clear", variant="secondary")
    
    # Connect components
    submit_btn.click(
        fn=voice_chat,
        inputs=[audio_input, chatbot],
        outputs=[audio_output, chatbot]
    )
    
    clear_btn.click(
        lambda: (None, []),
        inputs=None,
        outputs=[audio_output, chatbot]
    )

    gr.Markdown("""
    ### Instructions
    1. Click the microphone button and speak your question
    2. Click Submit to get a response
    3. The assistant will respond with voice and text
    """)

if __name__ == "__main__":
    demo.launch()
