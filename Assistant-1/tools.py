from datetime import datetime
import requests
import json
from utils.utilis import log
from rag import search 


def get_current_time():
    return datetime.now().strftime("%Y-%M-%D %H:%M:%S")


def get_city_weather(city:str):
    url = f"https://wttr.in/{city}?format=3"
    res = requests.get(url, timeout=10)
    res.raise_for_status() #This will raise HTTP error when somthing will wrong
    return res.text

def search_knowledge_base(query):
    results = search(query)
    log(f"📚 Knowledge base search : {query}")

    for r in results:
        log(f"{r['score']:.2f} - {r['source']} - #{r['chunk_id']}")

    blocks = []
    for r in results:
        source_name = r['source']
        chunk_id = r['chunk_id']
        chunk_text = r['text']

        header = f"[{source_name}] #{chunk_id}"
        block = f"{header}\n{chunk_text}"
        blocks.append(block)

    return "\n\n".join(blocks)

FUNCTION_NAME_TO_FUNCTION_MAP = {
    "get_current_time":get_current_time,
    "get_city_weather":get_city_weather,
    "search_knowledge_base":search_knowledge_base,
}

TOOLS = [
    {
        "type":"function",
        "function":{
            "name":"get_current_time",
            "description":"Returns the current date and time as a formated string",
            "parameters":{"type":"object","properties":{}}
        }
    },
    {
        "type":"function",
        "function":{
            "name":"get_city_weather",
            "description":"Returns the current weather of city",
            "parameters":{
                "type":"object",
                "properties":{
                   "city": {
                        "type":"string",
                        "description":"City to get the weather from function"
                    }
                },
                "required":["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description" : "Searches the Zorvex Technologies company knowledge base: leave, medical, HR, hiring, tech, incident "
                            "reports and travel expense policies. Use it for any question about company policies or documents. "
                            "Returns the most relevant passages with their source.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The user's question, rewritten as a short, self-contained search query.",
                    }
                },
                "required": ["query"],
            },
        },
    },
]

def run_tools(tool_call):
    fun_name = tool_call.function.name

    try:
        fun_arg = json.loads(tool_call.function.arguments)
        log(f"  {fun_name}({fun_arg})")
        fn = FUNCTION_NAME_TO_FUNCTION_MAP[fun_name]
        result = fn(**fun_arg)
    except Exception as e:
        result = f"Error while running function:{fun_name} : {e}"

    log(f"  {result}")
    return str(result)