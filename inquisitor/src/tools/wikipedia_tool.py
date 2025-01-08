"""
This module contains the Wikipedia tool for the Inquisitor project.
"""

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

LENGHT_OF_RESPONSE = 1000
TOOL_DESCRIPTION = """
A tool to explain things in text format. 
Use this tool if you think the user's asked concept is best explained dthrough text
"""

wiki_api_wrapper = WikipediaAPIWrapper(
    top_k_results=1, doc_content_chars_max=LENGHT_OF_RESPONSE
)
wikipedia = WikipediaQueryRun(
    description=TOOL_DESCRIPTION,
    api_wrapper=wiki_api_wrapper,
)


if __name__ == "__main__":
    print(wikipedia.invoke("What is a mobius strip?"))
