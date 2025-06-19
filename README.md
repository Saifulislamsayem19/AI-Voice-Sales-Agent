# AI Voice Sales Agent

A sophisticated AI-powered voice sales agent built with FastAPI and LangGraph that conducts intelligent sales conversations for educational course offerings. The agent uses OpenAI's GPT models and advanced conversation flow management to provide personalized sales experiences.

![image](https://github.com/user-attachments/assets/ed27ac44-a91b-404f-9a91-0b8c796d57b3)


## 🚀 Features

- **Intelligent Conversation Flow**: Uses LangGraph to manage complex sales conversation stages
- **Voice Integration**: Built-in speech-to-text (Whisper) and text-to-speech capabilities
- **Adaptive Sales Process**: Dynamically adjusts conversation based on customer responses and sentiment
- **Customer Profiling**: Builds detailed customer profiles during conversations
- **Multi-stage Sales Pipeline**: Handles introduction, qualification, pitching, objection handling, and closing
- **Real-time Sentiment Analysis**: Analyzes customer sentiment to guide conversation flow
- **Course Recommendation Engine**: Recommends appropriate courses based on customer profiles

## 🏗️ Architecture

The application uses LangGraph to create a state-driven conversation flow with the following stages:

1. **Introduction** - Initial greeting and rapport building
2. **Qualification** - Gathering customer information and needs
3. **Pitch** - Presenting relevant course offerings
4. **Objection Handling** - Addressing customer concerns professionally
5. **Closing** - Attempting to secure enrollment or next steps
6. **Follow-up** - Scheduling demos or sending additional information

## 📋 Prerequisites

- Python 3.8+
- OpenAI API key
- FastAPI
- LangGraph
- LangChain

## 🛠️ Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/voice-sales-agent.git
cd voice-sales-agent
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
Create a `.env` file in the root directory:
```env
OPENAI_API_KEY=your_openai_api_key_here
```

## 🚀 Usage

### Starting the Server

```bash
python main.py
```

The API will be available at `http://localhost:8000`

### API Documentation

Access the interactive API documentation at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 🔌 API Endpoints

### Core Sales Conversation

#### `POST /start-call`
Start a new sales call session
```json
{
  "phone_number": "+1234567890",
  "customer_name": "John Doe"
}
```

#### `POST /respond/{call_id}`
Send customer response and get agent reply
```json
{
  "message": "I'm interested in learning AI"
}
```

#### `GET /conversation/{call_id}`
Retrieve full conversation history

### Voice Features

#### `POST /transcribe`
Convert audio to text using OpenAI Whisper
- Upload audio file (WAV, MP3, M4A, etc.)

#### `POST /tts`
Convert text to speech
```json
{
  "text": "Hello, this is Sarah from TechEdu Academy"
}
```

### Health Check

#### `GET /health`
Check API status and active sessions

## 💼 Course Offerings

The agent promotes two main courses:

### AI Mastery Bootcamp
- **Duration**: 12 weeks
- **Price**: $299 (special) / $499 (regular)
- **Focus**: LLMs, Computer Vision, MLOps
- **Benefits**: Job placement assistance, certification, lifetime access

### Data Science Fundamentals
- **Duration**: 8 weeks
- **Price**: $249 (special) / $399 (regular)
- **Focus**: Python, SQL, Statistics, Tableau
- **Benefits**: Real-world projects, mentor support

## 🧠 Conversation Intelligence

### Customer Profiling
The agent automatically builds customer profiles including:
- Experience level (beginner, intermediate, experienced)
- Professional role
- Learning goals and needs
- Timeline preferences
- Budget and time concerns
- Interest level and sentiment

### Objection Handling
Intelligent responses to common objections:
- **Price concerns**: ROI emphasis, payment plans, value comparison
- **Time concerns**: Flexible learning options, minimal time commitment
- **Relevance concerns**: Beginner-friendly approach, job placement stats

## 🔧 Configuration

### Environment Variables
- `OPENAI_API_KEY`: Your OpenAI API key (required)

### Customizing Courses
Modify the `COURSE_INFO` dictionary in the code to add or update course offerings:

```python
COURSE_INFO = {
    "Your Course Name": {
        "duration": "X weeks",
        "regular_price": "$XXX",
        "special_price": "$XXX",
        "benefits": ["Benefit 1", "Benefit 2"],
        "target_audience": ["audience1", "audience2"]
    }
}
```

### Sample Conversation Flow
```
Agent: "Hello John! This is Sarah from TechEdu Academy..."
Customer: "Hi, I'm interested in learning AI for my career"
Agent: "That's great! What's your current experience with AI?"
Customer: "I'm a complete beginner"
Agent: "Perfect! Our AI Mastery Bootcamp is designed for beginners..."
```

## 🚨 Error Handling

The API includes comprehensive error handling:
- Invalid call IDs return 404
- Ended sessions return 400
- API errors return 500 with detailed messages
- Transcription/TTS errors are handled gracefully

## 🔒 Security Considerations

- Store API keys securely in environment variables
- Implement rate limiting for production use
- Add authentication for sensitive endpoints
- Validate all user inputs
- Use HTTPS in production

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Issues**: Report bugs via GitHub Issues
- **Discussions**: Use GitHub Discussions for questions
- **Documentation**: Check the `/docs` endpoint for API reference

## 🔮 Future Enhancements

- [ ] Multi-language support
- [ ] Advanced analytics dashboard
- [ ] CRM integration
- [ ] A/B testing for conversation flows
- [ ] Webhook support for external systems
- [ ] Advanced voice cloning options
- [ ] Real-time conversation monitoring
- [ ] Performance metrics and reporting

## 📈 Performance

- Handles concurrent sessions efficiently
- Sub-500ms response times for most operations
- Scalable architecture with stateless design
- Memory-efficient conversation storage

---

**Built with ❤️ using FastAPI, LangGraph, and OpenAI**
