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
        final_msg = None
        for step in range(MAX_STEPS):
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOLS
            )

            msg = response.choices[0].message

            if response.choices[0].finish_reason != "tool_calls":
                final_msg = msg
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

        if final_msg is None:
            # Ran out of steps and model never gave the final answer
            final_msg_content = "Sorry, I could not finish this within the step limit."
        else:
            final_msg_content = final_msg.content or ""

        messages.append({
            "role":"assistant",
            "content": final_msg_content
        })

        print("\nAssistance : ", final_msg_content)


if __name__ == "__main__":
    assistance_loop()