from google import genai
from dotenv import load_dotenv
import os
from fastapi import FastAPI
from pydantic import BaseModel

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
app = FastAPI()

class Question(BaseModel):
    content: str

@app.post("/output")
async def content_output(question: Question):
    response = await client.aio.models.generate_content(
        model="gemini-2.5-flash",
        contents=question.content
    )
    return {"response": response.text}

@app.get("/health")
async def health():
    return {"status": "ok"}