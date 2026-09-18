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

def resolve_set_score(player_a_games, player_b_games):
    high = max(player_a_games, player_b_games)
    low = min(player_a_games, player_b_games)
    margin = high - low
    if high == 6 and low == 6:
        return 'Tiebreak'
    if high == 7 and low == 6:
        return 'Win by Tiebreak'
    if high == 6 and margin >= 2:
        return 'Win'
    if high == 7 and low == 5:
        return 'Win'
    if high < 6 or (high == 6 and low == 5):
        return 'In Progress'
    return 'Invalid'

set_scoring_function = types.FunctionDeclaration(
    name='resolve_set_score',
    description="Given the number of games won by player A and B respectively, return who has won the set. If no winner return if the set is still in progress or if a tiebreak is needed or underway.",
    parameters_json_schema={
        "type": "object",
        "properties": {
            "player_a_games": {
                "type": "integer",
                "description": "Number of games won in a set by Player A e.g. 2."
            },
            "player_b_games": {
                "type": "integer",
                "description": "Number of games won in a set by Player B e.g. 0."
            }
        },
        "required": ['player_a_games', 'player_b_games']
    }
)

tool = types.Tool(function_declarations=[weather_function, set_scoring_function])

if __name__ == "__main__":
    contents = ["Is the match in London going to be delayed, and who's ahead in the set at 5-4?"]

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
        if function_call.name == weather_function.name:
            result = match_delay_risk_from_weather(**function_call.args)
        elif function_call.name == set_scoring_function.name:
            result = resolve_set_score(**function_call.args)
        else:
            result = {"success": False, "error": "unknown_tool", "message": f"No handler for {function_call.name}"}

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
        print(final_response.function_calls)
        print(final_response.text)
    else:
        print("[no tool called]")
        print(response.text)