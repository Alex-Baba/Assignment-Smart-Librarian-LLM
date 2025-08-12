import openai
import os
from dotenv import load_dotenv

load_dotenv()  # Load thy secrets from .env

api_key = os.getenv("OPENAI_API_KEY")


openai.api_key = api_key

client = openai.OpenAI(api_key=api_key)

response = client.chat.completions.create(
    model="gpt-4.1-nano",
    messages=[
        {"role": "user", "content": "What is the meaning of life?"}
    ]
)

print(response.choices[0].message.content)