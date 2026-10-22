import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from utils.utilis import log
from tools import TOOLS, run_tools

load_dotenv()

BASE_URL = os.getenv("BASE_URL")
MODEL = os.getenv("MODEL")
API_KEY = os.getenv("API_KEY")

client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY
)

messages = [{
    "role":"system",
    "content":"Act as a humbal person and respond always within 20 w ords"
}]  # context / memory

# Safety limits : max number of model rounds for one user query
MAX_STEPS=5

def assistance_loop():
    while True:
        user_input = input("\nYOU : ")

        if user_input.lower() in ["exit", "quit", "bye", "goodbye"]:
            break

        messages.append({
            "role":"user",
            "content":user_input
        })


        # Chained tool calling
        for step in range(MAX_STEPS):
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOLS
            )

            msg = response.choices[0].message

            if(response.choices[0].finish_reason != "tool_calls"):
                break

            log(f"Step : {step+1} Model requested {len(msg.tool_calls)} tool calls")

            messages.append(msg.model_dump())

            for tool_call in response.choices[0].message.tool_calls:
                result = run_tools(tool_call)
                messages.append({
                    "role":"tool",
                    "tool_call_id":tool_call.id,
                    "content":result
                })
        else:
            # Ran out of steps and model never gave the final answer
            fallback= "Sorry, I could not finish this within the step limit."
            messages.append({
                "role":"assistance",
                "content":fallback
            })
            continue


        messages.append({
            "role":"assistance",
            "content": msg.content
        })

        print("\nAssistance : ", (msg.content or ""))


if __name__ == "__main__":
    assistance_loop()