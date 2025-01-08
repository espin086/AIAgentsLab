"""
This module contains the Image Generator tool for the Inquisitor project.
"""

import argparse

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


def create_dalle_api_wrapper(
    model: str = "dall-e-3", size: str = "1792x1024"
) -> DallEAPIWrapper:
    """
    Initializes the DallE API Wrapper with the specified model and size.

    Args:
        model (str): The model to use for image generation.
        size (str): The size of the generated image.

    Returns:
        DallEAPIWrapper: Configured DallE API wrapper.
    """
    return DallEAPIWrapper(model=model, size=size)


def get_image_generator_tool(
    model: str = "dall-e-3", size: str = "1792x1024"
) -> OpenAIDALLEImageGenerationTool:
    """
    Creates and returns the Image Generator tool for Langchain.

    Args:
        model (str, optional): The model to use for image generation. Defaults to "dall-e-3".
        size (str, optional): The size of the generated image. Defaults to "1792x1024".

    Returns:
        OpenAIDALLEImageGenerationTool: Configured Image Generator tool.
    """
    dalle_api_wrapper = create_dalle_api_wrapper(model=model, size=size)
    dalle_tool = OpenAIDALLEImageGenerationTool(
        api_wrapper=dalle_api_wrapper, description=TOOL_DESCRIPTION
    )
    return dalle_tool


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Use the Image Generator tool to create images."
    )
    parser.add_argument("prompt", type=str, help="The prompt to generate an image.")
    parser.add_argument(
        "--model",
        type=str,
        default="dall-e-3",
        help="The model to use for image generation.",
    )
    parser.add_argument(
        "--size",
        type=str,
        default="1792x1024",
        help="The size of the generated image.",
    )
    args = parser.parse_args()

    dalle_instance = create_dalle_api_wrapper(model=args.model, size=args.size)

    dalle_tool_instance = OpenAIDALLEImageGenerationTool(
        api_wrapper=dalle_instance, description=TOOL_DESCRIPTION
    )

    output = dalle_tool_instance.invoke(args.prompt)
    print(output)
