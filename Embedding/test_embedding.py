from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY")
)

response = client.embeddings.create(
    model="nomic-embed-text",
    input="I am a good boy"
)

print(response.data[0].embedding)