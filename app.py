from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Literal
import openai
import os
import uuid
import base64
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, asdict
import uvicorn
import json
import io
from typing_extensions import TypedDict

# LangGraph imports
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Voice Sales Agent API with LangGraph", version="1.0.0")

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

# Initialize OpenAI client and LLM
openai_client = openai.OpenAI(api_key=api_key)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, api_key=api_key)

# Conversation stages
class ConversationStage(str, Enum):
    START = "start"
    INTRODUCTION = "introduction"
    QUALIFICATION = "qualification"
    PITCH = "pitch"
    OBJECTION_HANDLING = "objection_handling"
    CLOSING = "closing"
    FOLLOW_UP = "follow_up"
    ENDED = "ended"

# Data models
@dataclass
class CustomerProfile:
    name: str
    phone_number: str
    experience_level: Optional[str] = None
    needs: List[str] = None
    timeline: Optional[str] = None
    role: Optional[str] = None
    budget_concerns: bool = False
    time_concerns: bool = False
    interest_level: str = "neutral" 
    
    def __post_init__(self):
        if self.needs is None:
            self.needs = []

@dataclass
class ConversationMessage:
    speaker: str  # "agent" or "customer"
    message: str
    timestamp: datetime

# LangGraph State Definition
class AgentState(TypedDict):
    call_id: str
    customer: CustomerProfile
    messages: List[ConversationMessage]
    stage: ConversationStage
    qualification_count: int
    objections_handled: List[str]
    is_active: bool
    next_action: str
    sentiment: str
    customer_input: str
    agent_response: str

# Course information
COURSE_INFO = {
    "AI Mastery Bootcamp": {
        "duration": "12 weeks",
        "regular_price": "$499",
        "special_price": "$299",
        "benefits": [
            "Learn LLMs, Computer Vision, and MLOps",
            "Hands-on projects with real-world applications",
            "Job placement assistance with partner companies",
            "Certificate upon completion",
            "24/7 community support",
            "Lifetime access to course materials"
        ],
        "target_audience": ["beginners", "professionals", "career_changers"]
    },
    "Data Science Fundamentals": {
        "duration": "8 weeks",
        "regular_price": "$399",
        "special_price": "$249",
        "benefits": [
            "Python, SQL, and Statistics",
            "Data visualization with Tableau",
            "Real-world project portfolio",
            "Industry mentor support"
        ],
        "target_audience": ["beginners", "analysts"]
    }
}

class VoiceSalesAgent:
    def __init__(self):
        self.llm = llm
        self.graph = self._create_graph()
        
    def _create_graph(self) -> StateGraph:
        """Create the LangGraph workflow for sales conversation"""
        workflow = StateGraph(AgentState)
        
        # Add nodes for each conversation stage
        workflow.add_node("analyze_input", self.analyze_customer_input)
        workflow.add_node("introduction", self.handle_introduction)
        workflow.add_node("qualification", self.handle_qualification)
        workflow.add_node("pitch", self.handle_pitch)
        workflow.add_node("objection_handling", self.handle_objections)
        workflow.add_node("closing", self.handle_closing)
        workflow.add_node("follow_up", self.handle_follow_up)
        workflow.add_node("end_call", self.end_call)
        
        # Set entry point
        workflow.set_entry_point("analyze_input")
        
        # Add conditional edges based on conversation flow
        workflow.add_conditional_edges(
            "analyze_input",
            self.route_conversation,
            {
                "introduction": "introduction",
                "qualification": "qualification", 
                "pitch": "pitch",
                "objection_handling": "objection_handling",
                "closing": "closing",
                "follow_up": "follow_up",
                "end_call": "end_call"
            }
        )
        
        # Add edges from each node back to analyzer
        for node in ["introduction", "qualification", "pitch", "objection_handling", "closing", "follow_up"]:
            workflow.add_edge(node, END)
        
        workflow.add_edge("end_call", END)
        
        return workflow.compile()
    
    def analyze_customer_input(self, state: AgentState) -> AgentState:
        """Analyze customer input to determine sentiment and extract information"""
        customer_message = state.get("customer_input", "")
        
        if not customer_message:  # Initial call
            state["stage"] = ConversationStage.INTRODUCTION
            state["sentiment"] = "neutral"
            return state
        
        # Sentiment analysis prompt
        sentiment_prompt = ChatPromptTemplate.from_messages([
            ("system", """Analyze the customer's message and determine:
            1. Sentiment: positive, neutral, negative
            2. Interest level: high, medium, low
            3. Any objections mentioned: price, time, relevance, experience
            4. Information about their background/needs
            
            Return JSON format: {{"sentiment": "", "interest": "", "objections": [], "info_extracted": ""}}"""),
            ("human", f"Customer message: {customer_message}")
        ])
        
        try:
            response = self.llm.invoke(sentiment_prompt.format_messages())
            analysis = json.loads(response.content)
            
            state["sentiment"] = analysis.get("sentiment", "neutral")
            
            # Update customer profile based on extracted info
            if analysis.get("info_extracted"):
                state["customer"] = self.update_customer_profile(
                    state["customer"], 
                    customer_message, 
                    analysis["info_extracted"]
                )
            
            # Track objections
            if analysis.get("objections"):
                if "objections_handled" not in state:
                    state["objections_handled"] = []
                state["objections_handled"].extend(analysis["objections"])
                
        except Exception as e:
            print(f"Analysis error: {e}")
            state["sentiment"] = "neutral"
        
        return state
    
    # Updated route_conversation method
    def route_conversation(self, state: AgentState) -> str:
        """Determine next conversation stage based on current state"""
        current_stage = state.get("stage", ConversationStage.START)
        sentiment = state.get("sentiment", "neutral")
        customer_input = state.get("customer_input", "")
        
        # Check for end conversation signals
        end_signals = ["not interested", "no thanks", "stop calling", "not right now", "bye", "goodbye"]
        if any(signal in customer_input.lower() for signal in end_signals):
            return "end_call"
        
        # Stage progression logic - FIXED TO STOP AFTER TIMELINE
        if current_stage == ConversationStage.START or current_stage == ConversationStage.INTRODUCTION:
            if not customer_input:  # Initial greeting
                return "introduction"
            else:
                state["stage"] = ConversationStage.QUALIFICATION
                return "qualification"
                
        elif current_stage == ConversationStage.QUALIFICATION:
            qualification_count = state.get("qualification_count", 0)
            
            if qualification_count >= 2:
                state["stage"] = ConversationStage.PITCH
                return "pitch"
            
            # Otherwise, ask another qualification question
            return "qualification"
            
        elif current_stage == ConversationStage.PITCH:
            if "objection" in state or any(word in customer_input.lower() for word in ["expensive", "cost", "time", "busy"]):
                state["stage"] = ConversationStage.OBJECTION_HANDLING
                return "objection_handling"
            else:
                state["stage"] = ConversationStage.CLOSING
                return "closing"
                
        elif current_stage == ConversationStage.OBJECTION_HANDLING:
            # After handling objection, either close or pitch again
            if sentiment == "positive":
                state["stage"] = ConversationStage.CLOSING
                return "closing"
            else:
                # Try a different pitch approach
                state["stage"] = ConversationStage.PITCH
                return "pitch"
                
        elif current_stage == ConversationStage.CLOSING:
            if sentiment == "positive":
                state["stage"] = ConversationStage.FOLLOW_UP
                return "follow_up"
            else:
                return "end_call"
        
        elif current_stage == ConversationStage.FOLLOW_UP:
            # After follow-up, end the call
            return "end_call"
        
        return "end_call"
    
    def handle_introduction(self, state: AgentState) -> AgentState:
        """Handle introduction stage"""
        customer_name = state["customer"].name
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are Sarah, a friendly and professional AI sales agent for TechEdu Academy. 
            Create a warm, personalized introduction that:
            1. Greets the customer by name
            2. Briefly introduces yourself and TechEdu Academy
            3. Mentions calling about AI and tech training programs
            4. Asks if they have a moment to chat
            
            Keep it conversational, professional, and under 3 sentences. Sound human and natural."""),
            ("human", f"Create an introduction for {customer_name}")
        ])
        
        response = self.llm.invoke(prompt.format_messages())
        state["agent_response"] = response.content
        state["stage"] = ConversationStage.INTRODUCTION
        
        return state
    
    def handle_qualification(self, state: AgentState) -> AgentState:
        """Handle qualification questions"""
        customer = state["customer"]
        qualification_count = state.get("qualification_count", 0)
        
        # Determine what to ask based on what we know
        questions_to_ask = []
        if not customer.experience_level:
            questions_to_ask.append("experience")
        if not customer.role:
            questions_to_ask.append("role")
        if not customer.needs:
            questions_to_ask.append("goals")
        if not customer.timeline:
            questions_to_ask.append("timeline")
        
        question_type = questions_to_ask[0] if questions_to_ask else "goals"
        
        question_prompts = {
            "experience": f"Ask {customer.name} about their current experience with AI, machine learning, or data science. Keep it conversational.",
            "role": f"Ask {customer.name} about their current job role or what they do professionally. Make it sound natural.",
            "goals": f"Ask {customer.name} what they hope to achieve by learning AI or tech skills. Keep it friendly.",
            "timeline": f"Ask {customer.name} about their timeline for learning new skills or making a career change."
        }
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are Sarah from TechEdu Academy having a sales conversation. 
            Generate ONE natural qualifying question: {question_prompts.get(question_type, question_prompts['goals'])}
            
            Previous conversation context: {self._get_conversation_history(state)}
            
            Make it sound conversational and genuinely interested in helping them."""),
            ("human", "Generate the qualification question")
        ])
        
        response = self.llm.invoke(prompt.format_messages())
        state["agent_response"] = response.content
        state["qualification_count"] = qualification_count + 1
        
        return state
    
    def handle_pitch(self, state: AgentState) -> AgentState:
        """Handle course pitch based on customer profile"""
        customer = state["customer"]
        
        # Select best course based on customer profile
        best_course = self._select_best_course(customer)
        course_info = COURSE_INFO[best_course]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are Sarah from TechEdu Academy. Present the {best_course} course to {customer.name}.
            
            Customer profile: {asdict(customer)}
            Course details: {course_info}
            Conversation history: {self._get_conversation_history(state)}
            
            Create a personalized pitch that:
            1. Mentions the specific course name
            2. Highlights 2-3 benefits most relevant to their profile
            3. Mentions the special pricing
            4. Sounds natural and consultative, not pushy
            
            Keep it to 3-4 sentences maximum."""),
            ("human", "Create the personalized pitch")
        ])
        
        response = self.llm.invoke(prompt.format_messages())
        state["agent_response"] = response.content
        
        return state
    
    def handle_objections(self, state: AgentState) -> AgentState:
        """Handle customer objections professionally"""
        customer_input = state.get("customer_input", "")
        customer = state["customer"]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are Sarah from TechEdu Academy. {customer.name} has expressed a concern: "{customer_input}"
            
            Address their objection professionally using these guidelines:
            
            Price concerns: 
            - Emphasize ROI of AI skills (average 40% salary increase)
            - Mention payment plans available
            - Compare to cost of degree programs
            
            Time concerns:
            - Mention flexible, self-paced learning
            - Only 5-10 hours per week required
            - Weekend and evening options
            
            Relevance/Experience concerns:
            - Highlight beginner-friendly approach
            - Mention job placement assistance (85% placement rate)
            - Industry demand for AI skills
            
            Be empathetic, helpful, and solution-focused. Keep it to 2-3 sentences."""),
            ("human", "Address this objection professionally")
        ])
        
        response = self.llm.invoke(prompt.format_messages())
        state["agent_response"] = response.content
        
        return state
    
    def handle_closing(self, state: AgentState) -> AgentState:
        """Handle closing attempt"""
        customer = state["customer"]
        sentiment = state.get("sentiment", "neutral")
        
        if sentiment == "positive":
            close_type = "direct"
        else:
            close_type = "soft"
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are Sarah from TechEdu Academy closing the conversation with {customer.name}.
            
            Customer sentiment: {sentiment}
            Conversation history: {self._get_conversation_history(state)}
            
            Create a {close_type} close that:
            1. Summarizes key benefits relevant to them
            2. Creates appropriate urgency (limited-time offer)
            3. Offers specific next steps
            
            For positive sentiment: Ask for enrollment or demo scheduling
            For neutral/negative: Offer to send information or schedule follow-up call
            
            Keep it friendly and professional, not pushy."""),
            ("human", "Create the closing attempt")
        ])
        
        response = self.llm.invoke(prompt.format_messages())
        state["agent_response"] = response.content
        
        return state
    
    def handle_follow_up(self, state: AgentState) -> AgentState:
        """Handle follow-up scheduling"""
        customer = state["customer"]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are Sarah from TechEdu Academy. {customer.name} has shown interest.
            
            Provide follow-up options:
            1. Schedule a 15-minute demo call
            2. Send detailed course information via email
            3. Connect them with a program advisor
            
            Be helpful and make it easy for them to take the next step."""),
            ("human", "Provide follow-up options")
        ])
        
        response = self.llm.invoke(prompt.format_messages())
        state["agent_response"] = response.content
        
        return state
    
    def end_call(self, state: AgentState) -> AgentState:
        """End the call professionally"""
        customer = state["customer"]
        
        state["agent_response"] = f"I completely understand, {customer.name}. Thank you for taking the time to speak with me today. If you have any questions in the future about our programs, please don't hesitate to reach out. Have a wonderful day!"
        state["is_active"] = False
        state["stage"] = ConversationStage.ENDED
        
        return state
    
    def update_customer_profile(self, customer: CustomerProfile, message: str, extracted_info: str) -> CustomerProfile:
        """Update customer profile based on conversation"""
        message_lower = message.lower()
        
        # Experience level
        if any(word in message_lower for word in ["beginner", "new to", "never", "starting"]):
            customer.experience_level = "beginner"
        elif any(word in message_lower for word in ["experienced", "worked with", "years", "expert"]):
            customer.experience_level = "experienced"
        elif any(word in message_lower for word in ["some", "little", "basic"]):
            customer.experience_level = "intermediate"
        
        # Role detection
        roles = ["developer", "analyst", "manager", "student", "engineer", "consultant", "researcher"]
        for role in roles:
            if role in message_lower:
                customer.role = role
                break
        
        # Needs extraction
        if any(word in message_lower for word in ["career", "job", "promotion", "salary"]):
            if "career_advancement" not in customer.needs:
                customer.needs.append("career_advancement")
        if any(word in message_lower for word in ["business", "company", "startup"]):
            if "business_application" not in customer.needs:
                customer.needs.append("business_application")
        if any(word in message_lower for word in ["learn", "education", "knowledge", "skills"]):
            if "learning" not in customer.needs:
                customer.needs.append("learning")
        
        # Timeline
        if any(word in message_lower for word in ["soon", "quickly", "asap", "immediately"]):
            customer.timeline = "immediate"
        elif any(word in message_lower for word in ["month", "months", "few months"]):
            customer.timeline = "3-6 months"
        elif any(word in message_lower for word in ["year", "eventually", "future"]):
            customer.timeline = "6+ months"
        
        return customer
    
    def _select_best_course(self, customer: CustomerProfile) -> str:
        """Select the best course based on customer profile"""
        # Default to AI Mastery Bootcamp
        if customer.experience_level == "beginner":
            if "data" in " ".join(customer.needs).lower():
                return "Data Science Fundamentals"
        
        # For experienced users or those interested in AI specifically
        return "AI Mastery Bootcamp"
    
    def _get_conversation_history(self, state: AgentState) -> str:
        """Get formatted conversation history"""
        messages = state.get("messages", [])
        if not messages:
            return "No previous conversation"
        
        history = []
        for msg in messages[-6:]:  # Last 6 messages
            history.append(f"{msg.speaker}: {msg.message}")
        
        return "\n".join(history)
    
    async def process_conversation(self, state: AgentState) -> AgentState:
        """Process conversation through the graph"""
        result = await self.graph.ainvoke(state)
        return result

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

# In-memory storage
active_sessions: Dict[str, AgentState] = {}

# Initialize the sales agent
sales_agent = VoiceSalesAgent()

# API Endpoints
@app.post("/start-call", response_model=StartCallResponse)
async def start_call(request: StartCallRequest):
    """Start a new sales call session"""
    call_id = str(uuid.uuid4())
    
    # Create customer profile
    customer = CustomerProfile(
        name=request.customer_name,
        phone_number=request.phone_number
    )
    
    # Initialize state
    initial_state: AgentState = {
        "call_id": call_id,
        "customer": customer,
        "messages": [],
        "stage": ConversationStage.START,
        "qualification_count": 0,
        "objections_handled": [],
        "is_active": True,
        "next_action": "",
        "sentiment": "neutral",
        "customer_input": "",
        "agent_response": ""
    }
    
    # Process initial greeting
    result = await sales_agent.process_conversation(initial_state)
    greeting = result.get("agent_response", f"Hello {customer.name}! This is Sarah from TechEdu Academy. I'm calling about our exciting AI training programs. Do you have a moment to chat?")
    
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
        response = openai_client.audio.speech.create(
            model="tts-1",
            input=text,
            voice="nova"
        )
        
        # Convert the audio content to base64
        audio_content = response.content
        audio_base64 = base64.b64encode(audio_content).decode('utf-8')
        
        return JSONResponse(content={"audio": audio_base64})
    
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"TTS error: {str(e)}"
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