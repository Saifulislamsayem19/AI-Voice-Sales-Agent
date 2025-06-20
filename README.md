# Voice Sales Agent with LangGraph

A sophisticated AI-powered sales agent built with FastAPI and LangGraph that conducts natural voice conversations for TechEdu Academy's AI and Data Science training programs. The system combines speech recognition, intelligent conversation flow, and text-to-speech capabilities to create engaging sales interactions.

![image](https://github.com/user-attachments/assets/e93e1a7e-764a-47e5-a48c-131b00602eda)


## 🚀 Features

### Core Capabilities
- **Intelligent Conversation Flow**: Multi-stage sales process management using LangGraph
- **Real-time Voice Processing**: Audio transcription using OpenAI Whisper
- **Dual TTS Support**: HuggingFace SpeechT5 with OpenAI TTS fallback
- **Customer Profiling**: Dynamic customer qualification and needs assessment
- **Objection Handling**: AI-powered objection detection and response
- **Sentiment Analysis**: Real-time conversation sentiment tracking

### Sales Process Stages
1. **Introduction** - Warm greeting and company introduction
2. **Qualification** - Customer needs assessment and profiling
3. **Pitch** - Personalized course recommendations
4. **Objection Handling** - Address customer concerns
5. **Closing** - Conversion attempts based on customer sentiment
6. **Follow-up** - Professional call conclusion

### Technical Features
- **RESTful API** with FastAPI framework
- **Async Processing** for optimal performance
- **Session Management** for multiple concurrent calls
- **Error Handling** with graceful fallbacks
- **CORS Support** for web integration

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   FastAPI       │    │   LangGraph      │    │   AI Services   │
│   Web Server    │◄──►│   Workflow       │◄──►│   OpenAI GPT    │
│                 │    │   Engine         │    │   Whisper API   │
└─────────────────┘    └──────────────────┘    │   HuggingFace   │
         │                       │              │   SpeechT5      │
         │                       │              └─────────────────┘
         ▼                       ▼
┌─────────────────┐    ┌──────────────────┐
│   Session       │    │   Customer       │
│   Management    │    │   Profiling      │
└─────────────────┘    └──────────────────┘
```

### Key Components
- **VoiceSalesAgent**: Core AI agent with LangGraph workflow
- **AgentState**: Comprehensive conversation state management
- **CustomerProfile**: Dynamic customer data collection
- **ConversationStage**: Enum-based stage management

## 📋 Prerequisites

- Python 3.8+
- OpenAI API key
- CUDA-compatible GPU (optional, for faster TTS)

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/voice-sales-agent.git
   cd voice-sales-agent
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   # Create .env file
   echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
   ```

4. **Install PyTorch with CUDA support (optional)**
   ```bash
   # For CUDA 11.8
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

## 🚀 Usage

### Starting the Server

```bash
python app.py
```

The API will be available at `http://localhost:8000`

## 🎯 Course Offerings

The agent promotes two main courses:

### AI Mastery Bootcamp
- **Duration**: 12 weeks
- **Price**: $299 (Special) / $499 (Regular)
- **Target**: Professionals and experienced learners
- **Benefits**: LLMs, Computer Vision, MLOps, Job placement assistance

### Data Science Fundamentals
- **Duration**: 8 weeks
- **Price**: $249 (Special) / $399 (Regular)
- **Target**: Beginners and analysts
- **Benefits**: Python, SQL, Statistics, Tableau

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/start-call` | POST | Initialize new sales call session |
| `/respond/{call_id}` | POST | Process customer message and generate response |
| `/conversation/{call_id}` | GET | Retrieve full conversation history |
| `/transcribe` | POST | Convert audio to text using Whisper |
| `/tts` | POST | Convert text to speech with dual TTS support |
| `/health` | GET | System health check and metrics |

## 🔧 Configuration

### Environment Variables
```env
OPENAI_API_KEY=your_openai_api_key_here
```

### Model Configuration
- **LLM Model**: GPT-4o-mini (configurable)
- **TTS Primary**: HuggingFace SpeechT5
- **TTS Fallback**: OpenAI TTS (nova voice)
- **Transcription**: OpenAI Whisper-1

## 📈 Performance

- **Response Time**: < 2 seconds average
- **Concurrent Sessions**: 100+ supported
- **TTS Generation**: ~1 second for 50 words
- **Transcription**: ~3 seconds for 30-second audio

## 🔒 Security Features

- **Environment variable** configuration for API keys
- **CORS middleware** with configurable origins
- **Input validation** using Pydantic models
- **Error handling** with appropriate HTTP status codes
- **Session isolation** with unique call IDs

## 📊 Monitoring & Analytics

The system provides built-in analytics:

- **Active session tracking**
- **Conversation stage monitoring**
- **Customer profile evolution**
- **Objection pattern analysis**
- **Success rate metrics**

Access analytics via:
```http
GET /health  # Basic system health
GET /conversation/{call_id}  # Detailed conversation data
```

## 🔄 Development Workflow

### Adding New Conversation Stages

1. **Update the enum** in `ai_agent.py`:
```python
class ConversationStage(str, Enum):
    NEW_STAGE = "new_stage"
```

2. **Add the handler method**:
```python
def handle_new_stage(self, state: AgentState) -> AgentState:
    # Implementation here
    pass
```

3. **Update the graph workflow**:
```python
workflow.add_node("new_stage", self.handle_new_stage)
```

### Customizing System Prompts

Modify the `_get_system_prompts()` method in `VoiceSalesAgent` class to customize agent behavior for each stage.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🙏 Acknowledgments

- **OpenAI** for GPT and Whisper APIs
- **LangChain/LangGraph** for workflow orchestration
- **HuggingFace** for open-source TTS models
- **FastAPI** for the high-performance web framework

**Built with ❤️ OpenAI, LangGraph, HuggingFace, FastAPI  ** - Empowering careers through AI education
