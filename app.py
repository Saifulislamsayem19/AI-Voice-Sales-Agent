from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import openai
import os
import uuid
import base64
from datetime import datetime
import uvicorn
import io
from dataclasses import asdict
import numpy as np
import torch
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import scipy.io.wavfile as wav

# Import AI agent components
from ai_agent import (
    VoiceSalesAgent, 
    AgentState, 
    ConversationMessage, 
    create_initial_state
)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Voice Sales Agent API with Dual TTS Support", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load API key securely from environment variable
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set")

# Initialize OpenAI client
openai_client = openai.OpenAI(api_key=api_key)

# Initialize HuggingFace TTS Models (lazy loading)
class HuggingFaceTTS:
    def __init__(self):
        self.processor = None
        self.model = None
        self.vocoder = None
        self.speaker_embeddings = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._initialized = False
    
    def _initialize_models(self):
        """Initialize HuggingFace TTS models (lazy loading)"""
        if self._initialized:
            return
        
        try:
            print("Loading HuggingFace TTS models...")
            
            # Load SpeechT5 models
            self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
            self.model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
            self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
            
            # Move models to device
            self.model = self.model.to(self.device)
            self.vocoder = self.vocoder.to(self.device)
            
            # Load speaker embeddings
            embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
            self.speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0).to(self.device)
            
            self._initialized = True
            print("HuggingFace TTS models loaded successfully!")
            
        except Exception as e:
            print(f"Error loading HuggingFace TTS models: {str(e)}")
            raise e
    
    def generate_speech(self, text: str) -> bytes:
        """Generate speech from text using HuggingFace SpeechT5"""
        if not self._initialized:
            self._initialize_models()
        
        try:
            # Process text
            inputs = self.processor(text=text, return_tensors="pt").to(self.device)
            
            # Generate speech
            with torch.no_grad():
                speech = self.model.generate_speech(
                    inputs["input_ids"], 
                    self.speaker_embeddings, 
                    vocoder=self.vocoder
                )
            
            # Convert to numpy and then to audio bytes
            speech_np = speech.cpu().numpy()
            
            # Convert to 16-bit PCM audio
            audio_int16 = (speech_np * 32767).astype(np.int16)
            
            # Create WAV file in memory
            audio_buffer = io.BytesIO()
            wav.write(audio_buffer, 16000, audio_int16)  # 16kHz sample rate
            audio_buffer.seek(0)
            
            return audio_buffer.getvalue()
            
        except Exception as e:
            print(f"HuggingFace TTS Error: {str(e)}")
            raise e

# Initialize TTS systems
hf_tts = HuggingFaceTTS()

# Pydantic models for API
class StartCallRequest(BaseModel):
    phone_number: str
    customer_name: str

class StartCallResponse(BaseModel):
    call_id: str
    message: str
    first_message: str

class RespondRequest(BaseModel):
    message: str

class RespondResponse(BaseModel):
    reply: str
    should_end_call: bool
    stage: str
    customer_profile: Dict[str, Any]

class ConversationResponse(BaseModel):
    call_id: str
    history: List[Dict[str, Any]]
    customer_profile: Dict[str, Any]
    stage: str

class TranscribeRequest(BaseModel):
    audio_base64: str

class TTSRequest(BaseModel):
    text: str

class TTSResponse(BaseModel):
    audio: str  
    provider: str
    format: str

# In-memory storage
active_sessions: Dict[str, AgentState] = {}

# Initialize the sales agent
sales_agent = VoiceSalesAgent()

# API Endpoints
@app.post("/start-call", response_model=StartCallResponse)
async def start_call(request: StartCallRequest):
    """Start a new sales call session"""
    call_id = str(uuid.uuid4())
    
    # Create initial state
    initial_state = create_initial_state(call_id, request.customer_name, request.phone_number)
    
    # Process initial greeting
    result = await sales_agent.process_conversation(initial_state)
    greeting = result.get("agent_response", f"Hello {request.customer_name}! This is Sarah from TechEdu Academy. I'm calling about our exciting AI training programs. Do you have a moment to chat?")
    
    # Add greeting to conversation history
    greeting_message = ConversationMessage(
        speaker="agent",
        message=greeting,
        timestamp=datetime.now()
    )
    result["messages"] = [greeting_message]
    
    # Store session
    active_sessions[call_id] = result
    
    return StartCallResponse(
        call_id=call_id,
        message="Call started successfully",
        first_message=greeting
    )

@app.post("/respond/{call_id}", response_model=RespondResponse)
async def respond_to_customer(call_id: str, request: RespondRequest):
    """Generate agent response to customer message"""
    if call_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Call session not found")
    
    state = active_sessions[call_id]
    
    if not state.get("is_active", True):
        raise HTTPException(status_code=400, detail="Call session has ended")
    
    # Add customer message to conversation
    customer_message = ConversationMessage(
        speaker="customer",
        message=request.message,
        timestamp=datetime.now()
    )
    state["messages"].append(customer_message)
    
    # Update state with customer input
    state["customer_input"] = request.message
    
    # Process through graph
    result = await sales_agent.process_conversation(state)
    
    # Get agent response
    agent_reply = result.get("agent_response", "I'm sorry, could you repeat that?")
    should_end_call = not result.get("is_active", True)
    
    # Add agent response to conversation
    agent_message = ConversationMessage(
        speaker="agent",
        message=agent_reply,
        timestamp=datetime.now()
    )
    result["messages"].append(agent_message)
    
    # Update stored state
    active_sessions[call_id] = result
    
    return RespondResponse(
        reply=agent_reply,
        should_end_call=should_end_call,
        stage=result.get("stage", "unknown"),
        customer_profile=asdict(result["customer"])
    )

@app.get("/conversation/{call_id}", response_model=ConversationResponse)
async def get_conversation(call_id: str):
    """Get full conversation history"""
    if call_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Call session not found")
    
    state = active_sessions[call_id]
    
    history = [
        {
            "speaker": msg.speaker,
            "message": msg.message,
            "timestamp": msg.timestamp.isoformat()
        }
        for msg in state.get("messages", [])
    ]
    
    return ConversationResponse(
        call_id=call_id,
        history=history,
        customer_profile=asdict(state["customer"]),
        stage=state.get("stage", "unknown")
    )

@app.post("/transcribe")
async def transcribe_audio(audio_file: UploadFile = File(...)):
    """Transcribe audio to text using Whisper"""
    try:
        # Read the audio file
        audio_data = await audio_file.read()
        
        # Create a file-like object
        audio_file_like = io.BytesIO(audio_data)
        audio_file_like.name = "recording.wav"
        
        # Transcribe using Whisper API
        transcript = openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file_like
        )
        
        return JSONResponse(content={"text": transcript.text})
    
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Transcription error: {str(e)}"
        )

@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    """Convert text to speech using OpenAI TTS"""
    text = request.text
    
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")
    
    try:
        audio_content = hf_tts.generate_speech(text)
        audio_base64 = base64.b64encode(audio_content).decode('utf-8')
        
        return TTSResponse(
            audio=audio_base64,
            provider="huggingface",
            format="wav"
        )
    except Exception as hf_error:
        print(f"HuggingFace TTS failed: {str(hf_error)}")
        # Fallback to OpenAI TTS if HuggingFace fails
        print("Falling back to OpenAI TTS...")
        response = openai_client.audio.speech.create(
            model="tts-1",
            input=text,
            voice="nova"
        )
        audio_content = response.content
        audio_base64 = base64.b64encode(audio_content).decode('utf-8')
        
        return TTSResponse(
            audio=audio_base64,
            provider="openai_fallback",
            format="mp3"
        )

@app.get("/")
async def root():
    return {"message": "Voice Sales Agent API with LangGraph is running!"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "active_sessions": len(active_sessions),
        "langgraph_enabled": True
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
