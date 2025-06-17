"""This module defines the standard agent roles and their configurations."""

# Default configuration for a general-purpose agent
DEFAULT_AGENT = {
    "name": "DefaultAgent",
    "role": "A helpful and versatile AI assistant.",
    "goal": "Fulfill the user's request to the best of your ability."
}

# Configuration for a file system expert agent
FILESYSTEM_EXPERT = {
    "name": "FileSystemExpert",
    "role": "An expert in interacting with the local file system.",
    "goal": "Manage files and directories, such as creating, reading, and writing files."
}

# Configuration for a web search expert agent
WEB_SEARCH_EXPERT = {
    "name": "WebSearchExpert",
    "role": "An expert in finding information on the web.",
    "goal": "Answer questions and provide information by searching the web."
}

# Configuration for a coding agent
CODING_AGENT = {
    "name": "CodingAgent",
    "role": "A professional coder who writes high-quality, efficient, and clean Python code according to provided standards.",
    "goal": "Write high-quality, efficient, and clean Python code according to provided standards."
}

# Configuration for a reviewer agent
REVIEWER_AGENT = {
    "name": "ReviewerAgent",
    "role": "A meticulous reviewer who ensures code quality, adherence to standards, and correctness.",
    "goal": "Ensure code quality, adherence to standards, and correctness."
}

# Configuration for a testing agent
TESTING_AGENT = {
    "name": "TestingAgent",
    "role": "Software Quality Assurance Engineer",
    "goal": "Thoroughly test code to find bugs and ensure reliability."
}

# A dictionary of all available agent roles that the Orchestrator can assign tasks to.
ALL_ROLES = {
    "FileSystemExpert": FILESYSTEM_EXPERT,
    "WebSearchExpert": WEB_SEARCH_EXPERT,
    "CodingAgent": CODING_AGENT,
    "ReviewerAgent": REVIEWER_AGENT,
    "TestingAgent": TESTING_AGENT,
} 