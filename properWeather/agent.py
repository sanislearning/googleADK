#llms use the doc string as well in order to understand how and what a particular tool does

import logging.config
import os
import asyncio
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types #For creating message Content/Parts
import warnings
warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.ERROR)

from dotenv import load_dotenv
load_dotenv()

MODEL_GEMINI_2_0_FLASH="gemini-2.0-flash"

def get_weather(city:str)->dict:
    """Retrieves the current weather report for a specificed city
    Args:
    city(str): The name of the city (e.g., "New York","London","Tokyo").
    Returns:
    dict: A dictionary containing the weather information
        Includes a status key ('success' or 'error')
        If 'success', includes a 'report' key with the weather details.
        If 'error', includes an 'error_message' key.
    """
    print(f"---Tool: get_weather called for city{city}---")
    city_normalized=city.lower().replace(" ","") #converts all the city names to lower case and then replaces all of the spaces in them with nothing
    mock_weather_db={
    "newyork":{"status":"success","report":"The weather in New York is rainy with a temperature of 9°C"},
    "london":{"status":"success","report":"It's cloudy in London with a temperature of 15°C"},
    "tokyo":{"status":"success","report":"Tokyo is experiencing light rain and a temperature of 18°C."},
    }

    if city_normalized in mock_weather_db:
        return mock_weather_db[city_normalized]
    else:
        return {"status":"error","error_message":f"Sorry, I don't have weather information for {city}"}
    
#Agents in ADK orchestrates the interaction between the user, the LLM and the available tools.

AGENT_MODEL=MODEL_GEMINI_2_0_FLASH

weather_agent=Agent(
    name='weather_agent_v1',
    model=AGENT_MODEL,
    description="Provides weather information for specific cites.",
    instruction="You are a helpful weather agent,"
    "When the user asks for weather in a specific city,"
    "use the 'get_weather' tool to find the information."
    "If the tool returns an error, inform the user politely."
    "If the tool is successful, present the weather report clearly",
    tools=[get_weather]
)

#SessionService stores converstion history and state
#InMemorySessionService is simple, non-presistent storage

async def call_agent_async(query:str, runner,user_id,session_id):
    """Sends a query to the agent and prints the final response."""
    print(f"\n>>>User Query: {query}")
    #Prepare the user's message in ADK format
    content=types.Content(role="user",parts=[types.Part(text=query)])
    final_response_text="Agent did not produce a final response."
    #run_async executes the agent logic and yields Events.
    
    async for event in runner.run_async(user_id=user_id,session_id=session_id,new_message=content):
        #is_final_response() marks the concluding message for the turn
        if event.is_final_response():
            if event.content and event.content.parts:
                final_response_text=event.content.parts[0].text
            elif event.actions and event.actions.escalate: #Handles potential errors/escalations
                final_response_text=f'Agent escalated: {event.error_message or "No specific message."}'
            break #Stop processing events once the final response is found
    
    print(f"<<< Agent Response: {final_response_text}")

async def main():
    session_service=InMemorySessionService()

    #Define constants for identifying the interaction context
    APP_NAME="weather_tutorial_app"
    USER_ID="user_1"
    SESSION_ID="session_001"

    session=await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID
    )

    print(f"Session created: App='{APP_NAME}',user='{USER_ID}',Session='{SESSION_ID}'")

    #Runner
    #Runner orchestrates the agent execution loop.
    runner=Runner(
        agent=weather_agent, #The agent we want to run
        app_name=APP_NAME, #Associates runs with our app
        session_service=session_service #uses the session manager we made
    )
        
    print(f"Runner created for agent {runner.agent.name}.")

    await call_agent_async("What is the weather like in London?",
                           runner=runner,
                           user_id=USER_ID,
                           session_id=SESSION_ID)
    
    await call_agent_async("How about Paris?",
                           runner=runner,
                           user_id=USER_ID,
                           session_id=SESSION_ID)
    
    await call_agent_async("Tell me the weather in New York",
                           runner=runner,
                           user_id=USER_ID,
                           session_id=SESSION_ID)

#Interacting with the Agent
#We need to send messages to our agent and recieve responses. Since LLM calls and tool executions
#can take time, ADK's Runner operates asynchronously


"""What our call_agent_async function does:
    1. Takes a user query string.
    2. Packages it into the ADK Content format.
    3. Calls runner.run_async, providing the user/session context
    and new message.
    4. Iterates through the Events yielded by the runner. Events represent steps in the agent's execution.
    5. Identifies and prints the final response event using event.is_final_response()e"""


if __name__=="__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"An error occucred: {e}")