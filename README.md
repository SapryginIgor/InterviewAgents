# Multi-Agent Interview Coach

A multi-agent AI system that conducts technical interviews, featuring hidden reflection between agents and structured feedback generation.

## Architecture

The system implements a two-agent architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                    Interview Session                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌──────────────┐    Internal     ┌──────────────┐        │
│   │  OBSERVER    │◄───Guidance────►│  INTERVIEWER │        │
│   │   Agent      │                 │    Agent     │        │
│   │              │                 │              │        │
│   │ • Analyzes   │                 │ • Asks       │        │
│   │   responses  │                 │   questions  │        │
│   │ • Detects    │                 │ • Adapts     │        │
│   │   hallucin.  │                 │   difficulty │        │
│   │ • Suggests   │                 │ • Handles    │        │
│   │   next steps │                 │   off-topic  │        │
│   └──────────────┘                 └──────────────┘        │
│          │                                │                 │
│          │         Hidden from            │                 │
│          │         Candidate              │                 │
│          ▼                                ▼                 │
│   ┌─────────────────────────────────────────┐              │
│   │           Interview Logger               │              │
│   │  (Records visible + internal thoughts)   │              │
│   └─────────────────────────────────────────┘              │
│                        │                                    │
│                        ▼                                    │
│              interview_log.json                             │
└─────────────────────────────────────────────────────────────┘
```

## Features

### System Properties (as per specification)

1. **Role Specialization**: Two distinct agents with separate responsibilities
   - Observer: Analyzes responses, checks facts, detects hallucinations
   - Interviewer: Conducts dialogue, adapts questions

2. **Hidden Reflection**: Internal dialogue before each response
   - Observer analyzes candidate's answer
   - Provides guidance to Interviewer
   - All logged but hidden from candidate

3. **Context Awareness**: Full conversation history maintained
   - Agents remember previous exchanges
   - No repeated questions
   - Topic tracking

4. **Adaptability**: Dynamic difficulty adjustment
   - Questions get harder if candidate excels
   - Questions simplify if candidate struggles

5. **Robustness**: Handles edge cases
   - Off-topic detection and redirection
   - Hallucination detection and correction
   - Candidate questions are answered

### Feedback Structure

Final feedback includes:
- **Decision**: Assessed grade, hiring recommendation, confidence score
- **Technical Review**: Confirmed skills, knowledge gaps with correct answers
- **Soft Skills**: Clarity, honesty, engagement assessment
- **Personal Roadmap**: Specific topics to study

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd InterviewAgents

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your API key
```

## Configuration

Edit `.env` file:

```bash
# OpenAI API Key
OPENAI_API_KEY=sk-your-key-here

# Optional: Custom base URL (for proxies)
# OPENAI_BASE_URL=https://api.openai.com/v1

# Model (defaults to gpt-4o)
OPENAI_MODEL=gpt-4o
```

## Usage

### Interactive Mode

```bash
# With prompts for all inputs
python main.py

# With command-line arguments
python main.py --name "Alex" --position "Backend Developer" --grade "Junior" --experience "Django pet projects, some SQL"

# Hide internal thoughts (cleaner output)
python main.py --quiet

# Specify output file
python main.py --output interview_log.json
```

### Commands During Interview

- Type your responses normally
- Say **"стоп"** or **"stop"** to end the interview and get feedback

## Output Format

The system generates `interview_log_<name>_<timestamp>.json`:

```json
{
  "participant_name": "Alex",
  "position": "Backend Developer",
  "target_grade": "Junior",
  "experience": "Django pet projects",
  "timestamp": "2024-01-15T10:30:00",
  "turns": [
    {
      "turn_id": 1,
      "agent_visible_message": "Hello! Tell me about your experience...",
      "user_message": "Hi, I'm Alex, I know Python and SQL...",
      "internal_thoughts": "[Observer]: Candidate is a beginner. [Interviewer]: Will ask about basic data types."
    }
  ],
  "final_feedback": "..."
}
```

## Project Structure

```
InterviewAgents/
├── main.py                    # CLI entry point
├── requirements.txt           # Dependencies
├── .env.example              # Environment template
├── interview_coach/
│   ├── __init__.py
│   ├── llm.py                # OpenAI API client
│   ├── agents.py             # Interviewer & Observer agents
│   ├── manager.py            # Interview orchestration
│   ├── logger.py             # JSON logging
│   └── feedback.py           # Feedback generation
└── README.md
```

## Testing Scenarios

The system is designed to handle:

1. **Normal Q&A**: Standard technical questions and answers
2. **Hallucination Test**: False claims (e.g., "Python 4.0 will remove for loops")
3. **Off-topic**: Attempts to change the subject
4. **Role Reversal**: Candidate asking questions back
5. **Difficulty Adaptation**: Adjusting to candidate's level

## Extending the System

### Adding New Agents

Create a new agent class in `agents.py` following the pattern:

```python
class NewAgent:
    SYSTEM_PROMPT = "..."
    
    def __init__(self, llm: LLMClient):
        self.llm = llm
    
    def process(self, ...):
        # Agent logic
        pass
```

### Customizing Topics

Modify the `InterviewerAgent.SYSTEM_PROMPT` to focus on specific technologies or domains.

## License

MIT
