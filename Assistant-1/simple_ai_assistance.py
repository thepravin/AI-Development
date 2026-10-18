from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

BASE_URL = os.getenv("BASE_URL")
MODEL = os.getenv("MODEL")
API_KEY = os.getenv("API_KEY")

client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY
)

def assistance_loop():
    while True:
        user_input = input("\nYOU : ")

        if user_input.lower() in ["exit", "quit", "bye", "goodbye"]:
            break

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role":"user",
                    "content":user_input
                }
            ]
        )

        print("\nAssistance : ",response.choices[0].message.content)


if __name__ == "__main__":
    assistance_loop()