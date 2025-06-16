"""
This module defines the MemoryManager for the agent, responsible for
handling long-term memory storage and retrieval using SQLite.
"""
import sqlite3
import os
from typing import List, Dict, Any, Optional

class MemoryManager:
    """
    Manages the agent's long-term memory using a SQLite database.
    """
    def __init__(self, db_path: str = "db/memory.db"):
        """
        Initializes the MemoryManager.

        Args:
            db_path: The path to the SQLite database file.
        """
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.create_table()

    def create_table(self):
        """Creates the memory table if it doesn't exist."""
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS long_term_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fact TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

    def add_fact(self, fact: str) -> Dict[str, Any]:
        """
        Adds a new fact to the long-term memory.

        Args:
            fact: The piece of information to remember.
        
        Returns:
            A confirmation message.
        """
        if not fact or not isinstance(fact, str):
            return {"status": "error", "message": "Fact must be a non-empty string."}
        
        try:
            with self.conn:
                self.conn.execute(
                    "INSERT INTO long_term_memory (fact) VALUES (?)",
                    (fact,)
                )
            return {"status": "success", "message": f"Fact '{fact}' was successfully saved."}
        except sqlite3.Error as e:
            return {"status": "error", "message": f"Database error: {e}"}

    def get_recent_facts(self, limit: int = 10) -> List[str]:
        """
        Retrieves the most recent facts from memory.

        Args:
            limit: The maximum number of facts to retrieve.

        Returns:
            A list of recent facts.
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT fact FROM long_term_memory ORDER BY timestamp DESC LIMIT ?", (limit,))
            facts = [row[0] for row in cursor.fetchall()]
            return facts
        except sqlite3.Error:
            return []

    def close(self):
        """Closes the database connection."""
        if self.conn:
            self.conn.close() 