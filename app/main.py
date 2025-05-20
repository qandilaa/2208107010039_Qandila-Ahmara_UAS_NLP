from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import traceback
import uvicorn
import uuid
import shutil
from pathlib import Path

# Import functions from other modules
from app.stt import transcribe_speech_to_text
from app.llm import generate_response
from app.tts import transcribe_text_to_speech

app = FastAPI(title="Voice Chatbot API")

# Enable CORS to allow access from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create a directory for temporary audio files
AUDIO_DIR = Path("./temp_audio")
AUDIO_DIR.mkdir(exist_ok=True)

# Mount the audio directory as static files
app.mount("/audio", StaticFiles(directory=AUDIO_DIR), name="audio")

@app.post("/voice-chat")
async def voice_chat(file: UploadFile = File(...)):
    """
    Main endpoint to receive user audio, process it through STT → LLM → TTS pipeline,
    and return the response as audio with text transcriptions.
    """
    try:
        print(f"[INFO] File received: {file.filename}")

        # Read audio data from uploaded file
        audio_bytes = await file.read()
        ext = os.path.splitext(file.filename)[-1].lower()

        # Validate supported audio formats
        if ext not in ['.wav', '.mp3', '.m4a']:
            raise HTTPException(status_code=400, detail="Unsupported file type.")

        # Process Speech-to-Text (STT)
        text = transcribe_speech_to_text(audio_bytes, file_ext=ext)
        if text.startswith("[ERROR]"):
            raise HTTPException(status_code=500, detail=text)

        transcription = text.strip()
        print(f"[INFO] Transcription result: {transcription}")

        # Send text to LLM for response generation
        response = generate_response(transcription)
        if response.startswith("[ERROR]"):
            raise HTTPException(status_code=500, detail=response)

        response_text = response.strip()
        print(f"[INFO] Model response: {response_text}")

        # Convert response text to audio via TTS
        audio_path = transcribe_text_to_speech(response_text)
        if audio_path.startswith("[ERROR]"):
            raise HTTPException(status_code=500, detail=audio_path)

        # Copy the audio file to our static directory with unique name
        unique_filename = f"{uuid.uuid4()}.wav"
        output_path = AUDIO_DIR / unique_filename
        shutil.copy(audio_path, output_path)
        
        # Return JSON with transcription, response text, and audio URL
        return JSONResponse({
            "transcription": transcription,
            "response": response_text,
            "audio_url": f"/audio/{unique_filename}"
        })

    except Exception as e:
        print("[EXCEPTION]", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Voice Chatbot API is running. Use /voice-chat endpoint to interact."}

# Clean up function to delete old audio files (can be scheduled)
@app.get("/cleanup")
async def cleanup_audio_files(max_age_hours: int = 24):
    """Remove audio files older than specified hours"""
    import time
    current_time = time.time()
    deleted_count = 0
    
    for file_path in AUDIO_DIR.glob("*.wav"):
        file_age = current_time - os.path.getmtime(file_path)
        if file_age > (max_age_hours * 3600):  # Convert hours to seconds
            os.unlink(file_path)
            deleted_count += 1
    
    return {"message": f"Deleted {deleted_count} old audio files"}

# Run server if this file is executed directly
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

