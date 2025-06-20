from typing import List, Optional, Dict, Any, Literal
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, asdict
import json
import os

# LangGraph imports
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from typing_extensions import TypedDict

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Conversation Stage Enum
class ConversationStage(str, Enum):
    START = "start"
    INTRODUCTION = "introduction"
    QUALIFICATION = "qualification"
    PITCH = "pitch"
    OBJECTION_HANDLING = "objection_handling"
    CLOSING = "closing"
    END = "end"

# Customer Profile
@dataclass
class CustomerProfile:
    name: str
    phone_number: str
    needs: Optional[List[str]] = None
    experience_level: Optional[str] = None
    budget_concern: Optional[bool] = None
    time_availability: Optional[str] = None
    interested_course: Optional[str] = None
    objections: Optional[List[str]] = None
    qualification_responses: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.needs is None:
            self.needs = []
        if self.objections is None:
            self.objections = []
        if self.qualification_responses is None:
            self.qualification_responses = []

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

# Voice Sales Agent Class
class VoiceSalesAgent:
    def __init__(self):
        # Initialize OpenAI LLM
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, api_key=api_key)
        self.graph = self._create_graph()
    
    def _create_graph(self):
        """Create the LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Add nodes for each conversation stage
        workflow.add_node("analyze_input", self.analyze_input)
        workflow.add_node("introduction", self.handle_introduction)
        workflow.add_node("qualification", self.handle_qualification)
        workflow.add_node("pitch", self.handle_pitch)
        workflow.add_node("objection_handling", self.handle_objection)
        workflow.add_node("closing", self.handle_closing)
        workflow.add_node("end_call", self.end_call)
        
        # Set entry point
        workflow.set_entry_point("analyze_input")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "analyze_input",
            self.determine_next_stage,
            {
                ConversationStage.INTRODUCTION: "introduction",
                ConversationStage.QUALIFICATION: "qualification",
                ConversationStage.PITCH: "pitch",
                ConversationStage.OBJECTION_HANDLING: "objection_handling",
                ConversationStage.CLOSING: "closing",
                ConversationStage.END: "end_call"
            }
        )
        
        # Add edges from each stage
        workflow.add_edge("introduction", END)
        workflow.add_edge("qualification", END)
        workflow.add_edge("pitch", END)
        workflow.add_edge("objection_handling", END)
        workflow.add_edge("closing", END)
        workflow.add_edge("end_call", END)
        
        return workflow.compile()
    
    async def process_conversation(self, state: AgentState) -> AgentState:
        """Process the conversation through the graph"""
        result = await self.graph.ainvoke(state)
        return result
    
    def _prepare_conversation_history(self, state: AgentState) -> List[Any]:
        """Prepare conversation history for LLM"""
        messages = []
        
        for msg in state.get("messages", []):
            if msg.speaker == "agent":
                messages.append(AIMessage(content=msg.message))
            else:
                messages.append(HumanMessage(content=msg.message))
        
        return messages
    
    def _get_system_prompts(self) -> Dict[str, str]:
        """Get system prompts for each conversation stage"""
        return {
            "introduction": """You are Sarah, a friendly and professional sales agent from TechEdu Academy. 
        Your goal is to warmly greet the customer and introduce the company's AI training programs.
        Keep the introduction brief, friendly, and ask if they have a moment to chat.
        Company: TechEdu Academy specializes in AI and Data Science training programs.""",
            
            "qualification": """You are Sarah from TechEdu Academy. You're in the qualification stage.
        Ask thoughtful questions to understand the customer's:
        1. Current role and career goals
        2. Experience level with programming or data analysis
        3. Specific interests in AI/data science

        Ask one question at a time. Be conversational and show genuine interest in their responses.
        Based on their answers, mentally note if they're a beginner or experienced.""",
            
            "pitch": """You are Sarah from TechEdu Academy. Based on the customer's profile, present the most suitable course.
        Available courses:
        1. AI Mastery Bootcamp - 12 weeks, $299 (special price from $499) - For professionals and those with some experience
        2. Data Science Fundamentals - 8 weeks, $249 (special price from $399) - For beginners

        Highlight 3-4 key benefits relevant to their needs. Be enthusiastic but not pushy.
        End by asking what excites them most about the opportunity.""",
            
            "objection_handling": """You are Sarah from TechEdu Academy. Address the customer's concerns empathetically.
        Common objections and key points:
        - Price: Mention payment plans, ROI within 6 months, investment in future
        - Time: Flexible schedule, learn at own pace, designed for busy professionals
        - Experience: No prior experience needed, start from basics, mentor support
        - Relevance: AI skills essential across industries, future-proof career

        Be understanding and provide specific solutions. Don't be defensive.""",
            
            "closing": """You are Sarah from TechEdu Academy. Based on the conversation sentiment:
        - Positive: Offer to secure their spot today with special discount, or schedule with enrollment advisor
        - Neutral: Suggest sending detailed information via email or scheduling a follow-up call
        - Negative: Respectfully offer to send information for future consideration

        Always leave the door open and be professional. Provide clear next steps.""",
            
            "end": """You are Sarah from TechEdu Academy. End the call professionally.
        Thank them for their time, confirm any next steps discussed, and wish them well.
        Keep it brief and friendly."""
        }
    
    def _generate_llm_response(self, state: AgentState, stage: str, additional_context: str = "") -> str:
        """Generate response using LLM"""
        system_prompts = self._get_system_prompts()
        system_prompt = system_prompts.get(stage, "")
        
        # Add customer profile context
        customer_context = f"""
        Customer Profile:
        - Name: {state['customer'].name}
        - Experience Level: {state['customer'].experience_level or 'Unknown'}
        - Needs: {', '.join(state['customer'].needs) if state['customer'].needs else 'Not identified yet'}
        - Objections: {', '.join(state['customer'].objections) if state['customer'].objections else 'None'}
        - Qualification Responses: {'; '.join(state['customer'].qualification_responses) if state['customer'].qualification_responses else 'None yet'}
        """
        
        # Add stage-specific context
        if stage == "qualification":
            customer_context += f"\nQualification questions asked so far: {state.get('qualification_count', 0)}/3"
        elif stage == "pitch" and state['customer'].interested_course:
            course = state['customer'].interested_course
            course_info = COURSE_INFO[course]
            customer_context += f"\nRecommended course: {course}\nCourse details: {json.dumps(course_info, indent=2)}"
        elif stage == "objection_handling":
            customer_context += f"\nObjections raised: {', '.join(state['customer'].objections) if state['customer'].objections else 'None'}"
        elif stage == "closing":
            customer_context += f"\nClosing sentiment: {state.get('sentiment', 'neutral')}"

        # Combine prompts
        full_system_prompt = f"{system_prompt}\n\n{customer_context}\n\n{additional_context}"
        
        # Prepare messages
        messages = [SystemMessage(content=full_system_prompt)]
        
        # Add conversation history
        conv_history = self._prepare_conversation_history(state)
        messages.extend(conv_history)
        
        # Add latest customer input if available
        if state.get("customer_input"):
            messages.append(HumanMessage(content=state["customer_input"]))
        
        # Generate response
        try:
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            print(f"LLM Error: {str(e)}")
            return "I apologize, I'm having trouble processing that. Could you please repeat what you said?"
    
    async def analyze_input(self, state: AgentState) -> AgentState:
        """Analyze customer input and update state using LLM"""
        customer_input = state.get("customer_input", "")
        
        if not customer_input:
            return state
        
        # Use LLM to analyze sentiment and extract information
        analysis_prompt = f"""Analyze this customer response and extract:
        1. Sentiment (positive/neutral/negative)
        2. Any experience level indicators (beginner/professional/experienced)
        3. Any objections mentioned (price/time/relevance/experience)
        4. Any specific needs or interests

        Customer said: "{customer_input}"

        Respond in JSON format:
        {{"sentiment": "", "experience_level": "", "objections": [], "needs": []}}"""
        
        messages = [
            SystemMessage(content="You are an AI assistant analyzing customer responses for a sales call."),
            HumanMessage(content=analysis_prompt)
        ]
        
        try:
            analysis_response = self.llm.invoke(messages)
            analysis = json.loads(analysis_response.content)
            
            # Update state based on analysis
            state["sentiment"] = analysis.get("sentiment", "neutral")
            
            if analysis.get("experience_level"):
                state["customer"].experience_level = analysis["experience_level"]
            
            for objection in analysis.get("objections", []):
                if objection not in state["customer"].objections:
                    state["customer"].objections.append(objection)
            
            for need in analysis.get("needs", []):
                if need not in state["customer"].needs:
                    state["customer"].needs.append(need)
            
            # Store qualification response
            if state.get("stage") == ConversationStage.QUALIFICATION:
                state["customer"].qualification_responses.append(customer_input)
        
        except Exception as e:
            print(f"Analysis error: {str(e)}")
            # Fallback to simple keyword analysis
            customer_input_lower = customer_input.lower()
            if any(word in customer_input_lower for word in ["not interested", "busy", "no thanks", "goodbye"]):
                state["sentiment"] = "negative"
            elif any(word in customer_input_lower for word in ["yes", "sure", "interested", "tell me more"]):
                state["sentiment"] = "positive"
            else:
                state["sentiment"] = "neutral"
        
        return state
    
    def determine_next_stage(self, state: AgentState) -> str:
        """Determine the next conversation stage"""
        current_stage = state.get("stage", ConversationStage.START)
        sentiment = state.get("sentiment", "neutral")
        
        # Handle negative sentiment
        if sentiment == "negative" and current_stage != ConversationStage.OBJECTION_HANDLING:
            if state["customer"].objections:
                return ConversationStage.OBJECTION_HANDLING
            else:
                return ConversationStage.CLOSING
        
        # Normal flow progression
        if current_stage == ConversationStage.START:
            return ConversationStage.INTRODUCTION
        elif current_stage == ConversationStage.INTRODUCTION:
            return ConversationStage.QUALIFICATION
        elif current_stage == ConversationStage.QUALIFICATION:
            if state["qualification_count"] >= 3:
                return ConversationStage.PITCH
            else:
                return ConversationStage.QUALIFICATION
        elif current_stage == ConversationStage.PITCH:
            if state["customer"].objections:
                return ConversationStage.OBJECTION_HANDLING
            else:
                return ConversationStage.CLOSING
        elif current_stage == ConversationStage.OBJECTION_HANDLING:
            return ConversationStage.CLOSING
        else:
            return ConversationStage.END
    
    def handle_introduction(self, state: AgentState) -> AgentState:
        """Handle introduction stage using LLM"""
        context = "This is the first interaction" if not state["messages"] else "Customer has agreed to chat"
        
        response = self._generate_llm_response(state, "introduction", context)
        
        state["agent_response"] = response
        state["stage"] = ConversationStage.QUALIFICATION
        return state
    
    def handle_qualification(self, state: AgentState) -> AgentState:
        """Handle qualification stage using LLM"""
        qual_count = state.get("qualification_count", 0)
        
        if qual_count >= 3:
            context = "All qualification questions have been asked. Transition to pitch."
            state["stage"] = ConversationStage.PITCH
        else:
            context = f"Ask qualification question {qual_count + 1} of 3. Make it relevant to their previous responses."
            state["qualification_count"] = qual_count + 1
        
        response = self._generate_llm_response(state, "qualification", context)
        
        state["agent_response"] = response
        return state
    
    def handle_pitch(self, state: AgentState) -> AgentState:
        """Handle pitch stage using LLM"""
        # Determine best course based on profile
        customer = state["customer"]
        
        if customer.experience_level == "beginner" or any("career" in need for need in customer.needs):
            customer.interested_course = "Data Science Fundamentals"
        else:
            customer.interested_course = "AI Mastery Bootcamp"
        
        context = f"Present the {customer.interested_course} as the perfect fit based on their profile."
        
        response = self._generate_llm_response(state, "pitch", context)
        
        state["agent_response"] = response
        state["stage"] = ConversationStage.CLOSING
        return state
    
    def handle_objection(self, state: AgentState) -> AgentState:
        """Handle objections using LLM"""
        latest_objection = state["customer"].objections[-1] if state["customer"].objections else "general concern"
        
        context = f"Address the customer's {latest_objection} objection specifically and empathetically."
        
        response = self._generate_llm_response(state, "objection_handling", context)
        
        state["objections_handled"].append(latest_objection)
        state["agent_response"] = response
        state["stage"] = ConversationStage.CLOSING
        return state
    
    def handle_closing(self, state: AgentState) -> AgentState:
        """Handle closing stage using LLM"""
        sentiment = state.get("sentiment", "neutral")
        
        context = f"Customer sentiment is {sentiment}. Provide appropriate closing based on their engagement level."
        
        response = self._generate_llm_response(state, "closing", context)
        
        state["agent_response"] = response
        state["stage"] = ConversationStage.END
        return state
    
    def end_call(self, state: AgentState) -> AgentState:
        """End the call gracefully using LLM"""
        context = "End the call professionally based on what was discussed."
        
        response = self._generate_llm_response(state, "end", context)
        
        state["agent_response"] = response
        state["is_active"] = False
        return state

# Utility functions for creating initial state
def create_initial_state(call_id: str, customer_name: str, phone_number: str) -> AgentState:
    """Create initial agent state for a new call"""
    customer = CustomerProfile(
        name=customer_name,
        phone_number=phone_number
    )
    
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
    
    return initial_state