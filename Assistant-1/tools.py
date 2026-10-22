from datetime import datetime
import requests


def get_current_time():
    return datetime.now().strftime("%Y-%M-%D %H:%M:%S")


def get_city_weather(city:str):
    url = f"https://wttr.in/{city}?format=3"
    res = requests.get(url, timeout=10)
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
                }
            }
        }
    }
]