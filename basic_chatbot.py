from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

print("Gemini Chat - type 'quit' to exit")

while True:
    user_input = input("You: ")

    if not user_input.strip():
            print("Please type a question")
            continue
    
    if user_input.lower() == "quit":
        print("Goodbye!")
        break

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=user_input
        )
    except Exception as e:
        print(f'Error calling the API {e}')
        print("Retrying...")
        # retry logic
        try:  
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents={user_input}
            )
        except Exception as e:
            print(f'Retry failed: {e}')
            print("Please try again")
            continue

    print(response.text)


