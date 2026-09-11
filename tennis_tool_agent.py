import requests
from google import genai
from dotenv import load_dotenv
import os
from google.genai import types

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

def match_delay_risk_from_weather(location):
    try:
        # Geocode the location
        location_response = requests.get(
            'https://geocoding-api.open-meteo.com/v1/search',
            params={"name": location},
            timeout=5
        )

        location_response.raise_for_status()

        location_data = location_response.json()

        # Check that a location was actually found
        if not location_data.get("results"):
            return {
                "success": False,
                "error": "location_not_found",
                "message": f"Could not find a location for '{location}'."
            }

        location_result = location_data["results"][0]

        latitude = location_result["latitude"]
        longitude = location_result["longitude"]

        # Get weather for the location
        weather_response = requests.get(
            'https://api.open-meteo.com/v1/forecast',
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current": 'temperature_2m,precipitation,wind_speed_10m'
            },
            timeout=5
        )

        weather_response.raise_for_status()

        data = weather_response.json()

        return {
            "success": True,
            "location": {
                "name": location_result["name"],
                "country": location_result.get("country"),
                "latitude": latitude,
                "longitude": longitude
            },
            "data": data
        }

    except requests.Timeout:
        return {
            "success": False,
            "error": "weather_service_timeout",
            "message": "The weather service took too long to respond."
        }

    except requests.RequestException:
        return {
            "success": False,
            "error": "weather_service_unavailable",
            "message": "The weather service is currently unavailable."
        }

    except ValueError:
        return {
            "success": False,
            "error": "invalid_response",
            "message": "The weather service returned an invalid response."
        }

weather_function = types.FunctionDeclaration(
    name="match_delay_risk_from_weather",
    description="Given a location, return the current temperature, precipitation and wind speed so the model can judge whether an outdoor tennis match would be disrupted.",
    parameters_json_schema={
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city e.g. London."
            }
        },
        "required": ["location"]
    }
)

tool = types.Tool(function_declarations=[weather_function])

if __name__ == "__main__":
    contents = ["Will it rain during play in London today?"]

    response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=contents,
                config=types.GenerateContentConfig(
                    tools=[tool], 
                    system_instruction=("You are a tennis assistant. Only answer questions related to tennis e.g. "
                    "players, tournaments, rules, history, and match conditions. You have access to a live weather "
                    "tool, use it only when a tennis-relevant question needs current conditions to assess whether "
                    "an outdoor match would be disrupted. If a question is not about tennis, politely say so and "
                    "redirect the user back to tennis."
)
                )
            )

    function_call = response.function_calls[0] if response.function_calls else None

    if function_call:
        print(f"[tool called: {function_call.name}({function_call.args})]")
        result = match_delay_risk_from_weather(**function_call.args)

        function_response_part = types.Part.from_function_response(
            name=function_call.name,
            response={"result": result}
        )

        # append the model's turn (the function call) and your reply (the result) to the conversation
        contents.append(response.candidates[0].content)
        contents.append(types.Content(role="user", parts=[function_response_part]))

        final_response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=types.GenerateContentConfig(tools=[tool])
        )

        print(final_response.text)
    else:
        print("[no tool called]")
        print(response.text)