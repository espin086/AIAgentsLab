"""
Inquisitor Agent
A ReAct agent that explains topics via text, image, or video using Wikipedia, Dall-E, and YouTube search tools.
"""

import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

from .tools.wikipedia_tool import get_wikipedia_tool
from .tools.youtube_search_tool import get_youtube_search_tool
from .tools.image_generator_tool import get_image_generator_tool

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Define the system prompt
SYSTEM_PROMPT = """
   You are a helpful bot named Chandler. Your task is to explain topics
   asked by the user via three mediums: text, image or video.
  
   If the asked topic is best explained in text format, use the Wikipedia tool.
   If the topic is best explained by showing a picture of it, generate an image
   of the topic using Dall-E image generator and print the image URL.
   Finally, if video is the best medium to explain the topic, conduct a YouTube search on it
   and return found video links.
"""

# Setup tools and model
tools = [get_wikipedia_tool(), get_youtube_search_tool(), get_image_generator_tool()]

chat_model = ChatOpenAI(api_key=OPENAI_API_KEY)

# Create prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# Create agent
agent = create_openai_functions_agent(chat_model, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

# Run the agent
response = agent_executor.invoke(
    {"input": "explain to me photosynthesis", "chat_history": []}
)

print(response)
