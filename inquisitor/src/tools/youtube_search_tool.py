"""A tool to search YouTube videos."""

import argparse

from langchain_community.tools import YouTubeSearchTool

TOOL_DESCRIPTION = """
A tool to search YouTube videos. 
Use this tool if you think the user’s asked concept can be best explained by watching a video.
"""


def get_youtube_search_tool() -> YouTubeSearchTool:
    """
    Creates and returns the YouTube Search tool for Langchain.

    Returns:
        YouTubeSearchTool: Configured YouTube Search tool.
    """
    youtube_tool = YouTubeSearchTool(description=TOOL_DESCRIPTION)
    return youtube_tool


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Use the YouTube Search tool to find relevant videos."
    )
    parser.add_argument("query", type=str, help="The search query for YouTube.")
    args = parser.parse_args()

    youtube_tool_instance = get_youtube_search_tool()

    output = youtube_tool_instance.invoke(args.query)
    print(output)
