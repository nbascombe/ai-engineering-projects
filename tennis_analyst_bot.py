from google import genai
from dotenv import load_dotenv
import os
import json

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("Error: GOOGLE_API_KEY not found in .env file")
    exit(1)

client = genai.Client(api_key=api_key)

SYSTEM_PROMPT = """You are an expert tennis analyst with deep knowledge of tactics, history, and players.

You must ALWAYS respond with a valid JSON object in this exact format, no exceptions:
{
    "answer": "your full response here as a string",
    "players_mentioned": ["player1", "player2"],
    "related_topics": ["topic1", "topic2"],
    "is_tennis_related": true or false
}

Rules:
- answer: your full response. If not tennis-related, politely redirect to tennis here.
- players_mentioned: list of player full names mentioned. Empty list if none.
- related_topics: 2-3 relevant tennis topics (e.g. "Grand Slams", "Serve tactics"). Empty list if not tennis-related.
- is_tennis_related: true if the question is about tennis, false otherwise.

Return ONLY the JSON object. No preamble, no markdown, no code fences."""


def parse_response(raw_text):
    """Parse and validate the JSON response. Returns parsed dict or None."""
    try:
        # Strip markdown code fences if the model adds them despite instructions
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        
        parsed = json.loads(cleaned.strip())

        # Validate all required fields exist with correct types
        required = {
            "answer": str,
            "players_mentioned": list,
            "related_topics": list,
            "is_tennis_related": bool
        }
        for field, expected_type in required.items():
            if field not in parsed:
                raise ValueError(f"Missing field: {field}")
            if not isinstance(parsed[field], expected_type):
                raise ValueError(f"Wrong type for {field}: expected {expected_type.__name__}")

        return parsed

    except (json.JSONDecodeError, ValueError) as e:
        return None


def display_response(parsed):
    """Display a parsed response nicely in the terminal."""
    print()

    # Flag off-topic responses visually
    if not parsed["is_tennis_related"]:
        print("  [Off-topic]")

    print(f"  Analyst: {parsed['answer']}")

    if parsed["players_mentioned"]:
        players = ", ".join(parsed["players_mentioned"])
        print(f"\n  Players:  {players}")

    if parsed["related_topics"]:
        topics = " · ".join(parsed["related_topics"])
        print(f"  Topics:   {topics}")

    print()


def send_message(chat, user_input, retry=False):
    """Send a message and return a parsed response, with one retry on failure."""
    response = chat.send_message(user_input)
    raw = response.candidates[0].content.parts[0].text
    parsed = parse_response(raw)

    if parsed is None and not retry:
        # Retry once with an explicit nudge
        response = chat.send_message(
            "Your last response was not valid JSON. Please respond again using only the JSON format specified."
        )
        raw = response.candidates[0].content.parts[0].text
        parsed = parse_response(raw)

    return parsed, raw


try:
    chat = client.chats.create(
        model="gemini-2.5-flash",
        config={"system_instruction": SYSTEM_PROMPT}
    )
except Exception as e:
    print(f"Error creating chat session: {e}")
    exit(1)

print("Tennis Analysis Bot - type 'quit' to exit")
print("------------------------------------------")

while True:
    try:
        user_input = input("You: ")

        if not user_input.strip():
            print("Please type a question")
            continue

        if user_input.lower() == "quit":
            print("Goodbye!")
            break

        parsed, raw = send_message(chat, user_input)

        if parsed is not None:
            display_response(parsed)
        else:
            # Validation failed after retry — fall back to raw text
            print(f"\n  Analyst: {raw}\n")
            print("  [Note: structured parsing failed on this response]\n")

    except KeyboardInterrupt:
        print("\nGoodbye!")
        break
    except Exception as e:
        print(f"Something went wrong: {e}")
        print("Please try again")