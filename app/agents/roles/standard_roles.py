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

# Configuration for the planner agent
PLANNER_AGENT = {
    "name": "PlannerAgent",
    "role": "A master planner who specializes in breaking down complex goals into a sequence of actionable steps for a team of specialized agents.",
    "goal": "To create a clear, step-by-step JSON plan that efficiently leads to the user's desired outcome.",
    "tools": [] # The planner does not use tools, it only thinks.
}

# Configuration for the evaluator agent
EVALUATOR_AGENT = {
    "name": "EvaluatorAgent",
    "role": "A meticulous evaluator who analyzes multiple execution plans and selects the most optimal one.",
    "goal": "To choose the most efficient, logical, and safe plan from a given set of options.",
    "tools": [] # The evaluator only thinks and chooses.
} 