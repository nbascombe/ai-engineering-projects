from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

# Check API key exists before doing anything
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("Error: GOOGLE_API_KEY not found in .env file")
    exit(1)

client = genai.Client(api_key=api_key)

SYSTEM_PROMPT = """You are an expert tennis analyst with deep knowledge 
of tactics, history, and players. Answer questions in a clear, engaging way. 
If asked about something unrelated to tennis, politely redirect 
the conversation back to tennis."""

try:
    chat = client.chats.create(
        model="gemini-2.5-flash",
        config={
            "system_instruction": SYSTEM_PROMPT
        }
    )
except Exception as e:
    print(f"Error creating chat session: {e}")
    exit(1)

print("Tennis Analysis Bot - type 'quit' to exit")
print("------------------------------------------")

while True:
    try:
        user_input = input("You: ")

        # Handle empty input
        if not user_input.strip():
            print("Please type a question")
            continue

        if user_input.lower() == "quit":
            print("Goodbye!")
            break

        response = chat.send_message(user_input)
        print(f"Analyst: {response.candidates[0].content.parts[0].text}")
        print()

    except KeyboardInterrupt:
        # Handles Ctrl+C gracefully
        print("\nGoodbye!")
        break
    except Exception as e:
        print(f"Something went wrong: {e}")
        print("Please try again")