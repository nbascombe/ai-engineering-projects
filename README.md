# AI Engineering Projects

A collection of projects built while developing hands-on AI engineering skills.
Each project is intentional - focused on understanding the fundamentals before 
adding complexity.

---

## Projects

### 1. Basic Chatbot (`basic_chatbot.py`)
A simple single-turn chatbot that makes a direct call to the Gemini API and 
returns a response. The foundation for understanding how LLM API calls work.

**Concepts covered:**
- Gemini API integration
- Secure API key management with environment variables
- Stateless LLM calls

---

### 2. Tennis Analyst Bot (`tennis_analyst_bot.py`)
A conversational AI tennis analyst that maintains context across multiple 
messages, responds within a defined persona, and returns structured JSON on 
every response.

**Concepts covered:**
- Stateful conversation management with chat sessions
- System prompts to control model behaviour and personality
- Structured outputs - forcing the model to return a consistent JSON schema
- JSON validation and schema enforcement with typed field checking
- Retry on parse failure, fall back to raw text
- The difference between stateless calls and conversational memory

**Response schema:**
```json
{
    "answer": "string",
    "players_mentioned": ["array of strings"],
    "related_topics": ["array of strings"],
    "is_tennis_related": true
}
```

**Why structured outputs matter:**
At scale, downstream code reads LLM responses - not humans. Free text breaks 
parsers unpredictably. A consistent schema means reliable parsing, structured 
logging, and code that can make decisions based on response fields.

---

## Technical Progression
- `basic_chatbot.py` - stateless, single call, no memory
- `tennis_analyst_bot.py` - stateful, conversational, structured JSON outputs, validated responses

---

## Tech Stack
- Python 3.13
- Google Gemini 2.5 Flash
- google-genai
- python-dotenv

---

## Setup

1. Clone the repo
```
git clone https://github.com/nbascombe/ai-engineering-projects.git
```

2. Create and activate a virtual environment
```
python3.13 -m venv venv
source venv/bin/activate
```

3. Install dependencies
```
pip install -r requirements.txt
```

4. Create a `.env` file in the root folder
```
GOOGLE_API_KEY=your-key-here
```

5. Run a project
```
python basic_chatbot.py
python tennis_analyst_bot.py
```

---

*Nikita - AI Engineering Projects - 2026*