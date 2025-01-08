"""A tool to search YouTube videos."""


from langchain_community.tools import YouTubeSearchTool

TOOL_DESCRIPTION = """
A tool to search YouTube videos. 
Use this tool if you think the user’s asked concept can be best explained by watching a video.
"""


youtube = YouTubeSearchTool(description=TOOL_DESCRIPTION)

if __name__ == "__main__":
    print(youtube.invoke("What is a mobius strip?"))
