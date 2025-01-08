"""
This module contains the Image Generator tool for the Inquisitor project.
"""

from langchain_community.tools.openai_dalle_image_generation import (
    OpenAIDALLEImageGenerationTool,
)
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper

from dotenv import load_dotenv

load_dotenv()


TOOL_DESCRIPTION = """
A tool to generate images. 
Use this tool if you think the user's asked concept can be best explained by an image.
"""

dalle_api_wrapper = DallEAPIWrapper(model="dall-e-3", size="1792x1024")
dalle = OpenAIDALLEImageGenerationTool(
    api_wrapper=dalle_api_wrapper, description=TOOL_DESCRIPTION
)
output = dalle.invoke("A mountain bike illustration.")
print(output)
