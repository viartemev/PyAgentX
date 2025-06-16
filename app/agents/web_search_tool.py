"""
This module defines the web search tool for the agent.
"""
from typing import Dict, Any, List
from duckduckgo_search import DDGS

# 1. Tool Definition (JSON Schema)
web_search_tool_def = {
    "type": "function",
    "function": {
        "name": "web_search_tool",
        "description": "Searches the web for up-to-date information on a given topic. "
                       "Use this to find current events, facts, or information not present in the knowledge base.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to send to the search engine."
                }
            },
            "required": ["query"],
        },
    },
}

# 2. Tool Function
def web_search_tool(args: Dict[str, Any]) -> str:
    """
    Performs a web search using the DuckDuckGo search engine.

    Args:
        args: A dictionary containing the 'query' for the search.

    Returns:
        A formatted string containing the top search results, or an error message.
    """
    query = args.get("query")
    if not query:
        return "Error: The 'query' parameter is required for web_search_tool."

    try:
        with DDGS() as ddgs:
            # max_results=5 to keep the context concise
            results: List[Dict[str, str]] = list(ddgs.text(query, max_results=5))
            if not results:
                return f"No results found for '{query}'."

            # Format the results for the LLM
            formatted_results = []
            for i, result in enumerate(results):
                formatted_results.append(
                    f"Result {i+1}:\n"
                    f"  Title: {result.get('title')}\n"
                    f"  Snippet: {result.get('body')}\n"
                    f"  URL: {result.get('href')}\n"
                )
            return "\n---\n".join(formatted_results)
    except Exception as e:
        return f"Error during web search for '{query}': {e}" 