from datetime import datetime
import requests
import json
from utils.utilis import log


def get_current_time():
    return datetime.now().strftime("%Y-%M-%D %H:%M:%S")


def get_city_weather(city:str):
    url = f"https://wttr.in/{city}?format=3"
    res = requests.get(url, timeout=10)
    res.raise_for_status() #This will raise HTTP error when somthing will wrong
    return res.text


FUNCTION_NAME_TO_FUNCTION_MAP = {
    "get_current_time":get_current_time,
    "get_city_weather":get_city_weather
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
    }
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