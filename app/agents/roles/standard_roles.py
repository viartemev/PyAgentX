"""
This module defines the standard roles for specialized agents in the system.
"""

# Configuration for an agent specialized in file system operations
FILESYSTEM_EXPERT = {
    "name": "FileSystemExpert",
    "role": "An expert in browsing and reading files on a local file system.",
    "goal": "To help users understand the project structure by listing and reading files.",
    "tools": ["list_files", "read_file", "save_memory"]
}

# Configuration for an agent specialized in web searching
WEB_SEARCH_EXPERT = {
    "name": "WebSearchExpert",
    "role": "An expert in searching the web for real-time information.",
    "goal": "To find the most relevant and up-to-date information online in response to a user's query.",
    "tools": ["web_search", "save_memory"]
}

# Add other specialized agent configurations here as needed.
# For example, a CodeWriterAgent, a DatabaseExpert, etc.

ALL_ROLES = {
    "FileSystemExpert": FILESYSTEM_EXPERT,
    "WebSearchExpert": WEB_SEARCH_EXPERT,
} 