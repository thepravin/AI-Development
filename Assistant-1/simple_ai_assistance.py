import os
import json
from types import SimpleNamespace
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
    "content":(
        "You are a helpful, humble assistant."
        "Answer general questions (programming, science, advice) from your own knowledge."
        "Use a tool ONLY when the user asks for the current time, the weather, or their city."
        "Keep answers under 100 words and don't need to provide the number of words used until user ask."
    )
}]  # context / memory

# Safety limits : max number of model rounds for one user query
MAX_STEPS=5



def build_tool_calls_payload(tool_calls):
    '''
    Converts our collected tool calls into the dict format that the API expects
    inside an assistant message.
    '''
    payload = []
    for tool_call in tool_calls:
        tool_call_dict = {
            "id":tool_call.id,
            "type":"function",
            "function":{
                "name":tool_call.function.name,
                "arguments":tool_call.function.arguments,
            },
        }

        payload.append(tool_call_dict)

    return payload


def stream_chat(history):
    stream = client.chat.completions.create(
                model=MODEL,
                messages=history,
                tools=TOOLS,
                stream=True
            )

    full_response = ""
    calls={}
    assistance_label_printed = False

    # Streaming started
    try:
        for chunk in stream:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            #Do we have message (content) from LLM
            if delta.content:
                if not assistance_label_printed:
                    print("🤖 :", end="", flush=True)
                    assistance_label_printed = True
                print(delta.content, end="", flush=True)
                full_response += delta.content

            # Tool calls
            for tc in (delta.tool_calls or []):
                obj = calls.setdefault(
                    tc.index,
                    SimpleNamespace(id=None, function=SimpleNamespace(
                        name="",
                        arguments=""
                    )),                
                )

                if tc.id:
                    obj.id = tc.id
                if tc.function and tc.function.name:
                    obj.function.name += tc.function.name
                if tc.function and tc.function.arguments:
                    obj.function.arguments += tc.function.arguments
    except KeyboardInterrupt:
        print("\n BYE 👋👋👋")


    
    if assistance_label_printed:
        print()

    return full_response, list(calls)







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
            full_response, tool_calls = stream_chat(messages)

            # msg = response.choices[0].message

            if not tool_calls:
                final_msg = full_response
                break

            log(f"Step : {step+1} Model requested {len(tool_calls)} tool calls")

            messages.append({
                "role":"assistant",
                "content":full_response,
                "tool_calls":build_tool_calls_payload(tool_calls)
            })

            for tool_call in tool_calls:
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
            final_msg_content = final_msg or ""
            messages.append({"role":"assistant","content":final_msg_content})




if __name__ == "__main__":
    assistance_loop()