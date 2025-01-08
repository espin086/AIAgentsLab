"""
This module contains the Wikipedia tool for the Inquisitor project.
"""

import argparse

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper


LENGTH_OF_RESPONSE = 1000
TOOL_DESCRIPTION = """
A tool to explain things in text format. 
Use this tool if you think the user's asked concept is best explained through text.
"""


def create_wiki_api_wrapper(max_chars: int) -> WikipediaAPIWrapper:
    """
    Initializes the Wikipedia API Wrapper with the specified maximum characters.

    Args:
        max_chars (int): Maximum number of characters for document content.

    Returns:
        WikipediaAPIWrapper: Configured Wikipedia API wrapper.
    """
    return WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=max_chars)


def get_wikipedia_tool() -> WikipediaQueryRun:
    """
    Creates and returns the Wikipedia tool for Langchain.

    Returns:
        WikipediaQueryRun: Configured Wikipedia tool.
    """
    wiki_api_wrapper = WikipediaAPIWrapper(
        top_k_results=1, doc_content_chars_max=LENGTH_OF_RESPONSE
    )
    wikipedia = WikipediaQueryRun(
        description=TOOL_DESCRIPTION,
        api_wrapper=wiki_api_wrapper,
    )
    return wikipedia


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Use the Wikipedia tool to explain concepts."
    )
    parser.add_argument("query", type=str, help="The query to search on Wikipedia.")
    parser.add_argument(
        "--max_chars",
        type=int,
        default=LENGTH_OF_RESPONSE,
        help="Maximum number of characters for document content.",
    )
    args = parser.parse_args()

    wiki_api_instance = create_wiki_api_wrapper(args.max_chars)

    wikipedia_tool_instance = WikipediaQueryRun(
        description=TOOL_DESCRIPTION,
        api_wrapper=wiki_api_instance,
    )

    print(wikipedia_tool_instance.invoke(args.query))
