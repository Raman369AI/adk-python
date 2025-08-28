# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import json
import logging
from pathlib import Path
import re
from typing import TYPE_CHECKING

import aiosqlite
from google.genai import types
from typing_extensions import override

from . import _utils
from .base_memory_service import BaseMemoryService
from .base_memory_service import SearchMemoryResponse
from .memory_entry import MemoryEntry

if TYPE_CHECKING:
  from ..sessions.session import Session

logger = logging.getLogger('google_adk.' + __name__)


def _extract_words_lower(text: str) -> set[str]:
  """Extracts words from a string and converts them to lowercase."""
  return set([word.lower() for word in re.findall(r'[A-Za-z]+', text)])


class SqliteMemoryService(BaseMemoryService):
  """An async SQLite-based memory service for persistent storage with keyword search.

  This implementation provides persistent storage of memory entries in a SQLite
  database using aiosqlite for async operations while maintaining simple
  keyword-based search functionality similar to InMemoryMemoryService.

  This service is suitable for development and small-scale production use where
  semantic search is not required.
  """

  def __init__(self, db_path: str = 'memory.db'):
    """Initializes a SqliteMemoryService.

    Args:
        db_path: Path to the SQLite database file. Defaults to 'memory.db'
            in the current directory.
    """
    self._db_path = Path(db_path)

  async def _init_db(self):
    """Initializes the SQLite database with required tables."""
    # Create directory if it doesn't exist
    self._db_path.parent.mkdir(parents=True, exist_ok=True)

    async with aiosqlite.connect(self._db_path) as db:
      # Create memory_entries table
      await db.execute("""
          CREATE TABLE IF NOT EXISTS memory_entries (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              app_name TEXT NOT NULL,
              user_id TEXT NOT NULL,
              session_id TEXT NOT NULL,
              author TEXT,
              timestamp REAL NOT NULL,
              content_json TEXT NOT NULL,
              text_content TEXT NOT NULL,
              created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
              UNIQUE(app_name, user_id, session_id, timestamp, author)
          )
      """)

      # Create index for faster searches
      await db.execute("""
          CREATE INDEX IF NOT EXISTS idx_memory_search 
          ON memory_entries(app_name, user_id, text_content)
      """)

      # Create index for timestamp ordering
      await db.execute("""
          CREATE INDEX IF NOT EXISTS idx_memory_timestamp 
          ON memory_entries(timestamp DESC)
      """)

      await db.commit()

  @override
  async def add_session_to_memory(self, session: Session):
    """Adds a session to the SQLite memory store.

    A session may be added multiple times during its lifetime. Duplicate
    entries are ignored based on unique constraints.

    Args:
        session: The session to add to memory.
    """
    await self._init_db()

    async with aiosqlite.connect(self._db_path) as db:
      for event in session.events:
        if not event.content or not event.content.parts:
          continue

        # Extract text content for search
        text_parts = [part.text for part in event.content.parts if part.text]
        if not text_parts:
          continue

        text_content = ' '.join(text_parts)
        content_json = json.dumps(
            event.content.model_dump(exclude_none=True, mode='json')
        )

        try:
          await db.execute(
              """
              INSERT OR IGNORE INTO memory_entries 
              (app_name, user_id, session_id, author, timestamp, 
               content_json, text_content)
              VALUES (?, ?, ?, ?, ?, ?, ?)
          """,
              (
                  session.app_name,
                  session.user_id,
                  session.id,
                  event.author,
                  event.timestamp,
                  content_json,
                  text_content,
              ),
          )
        except Exception as e:
          logger.error(f'Error adding memory entry: {e}')
          continue

      await db.commit()
      logger.info(
          f'Added session {session.id} to memory for user {session.user_id}'
      )

  @override
  async def search_memory(
      self, *, app_name: str, user_id: str, query: str
  ) -> SearchMemoryResponse:
    """Searches for memories that match the query using keyword matching.

    Args:
        app_name: The name of the application.
        user_id: The id of the user.
        query: The query to search for.

    Returns:
        A SearchMemoryResponse containing matching memories ordered by timestamp.
    """
    await self._init_db()

    words_in_query = _extract_words_lower(query)

    async with aiosqlite.connect(self._db_path) as db:
      # Get all entries for the user
      async with db.execute(
          """
          SELECT author, timestamp, content_json, text_content
          FROM memory_entries 
          WHERE app_name = ? AND user_id = ?
          ORDER BY timestamp DESC
      """,
          (app_name, user_id),
      ) as cursor:

        rows = await cursor.fetchall()

    memories = []
    for author, timestamp, content_json, text_content in rows:
      words_in_entry = _extract_words_lower(text_content)

      # Check if any query words match entry words
      if any(query_word in words_in_entry for query_word in words_in_query):
        try:
          content_dict = json.loads(content_json)
          content = types.Content(**content_dict)

          memory_entry = MemoryEntry(
              content=content,
              author=author,
              timestamp=_utils.format_timestamp(timestamp),
          )
          memories.append(memory_entry)
        except (json.JSONDecodeError, ValueError) as e:
          logger.error(f'Error parsing memory entry content: {e}')
          continue

    return SearchMemoryResponse(memories=memories)

  async def clear_memory(self, app_name: str = None, user_id: str = None):
    """Clears memory entries from the database.

    Args:
        app_name: If specified, only clear entries for this app.
        user_id: If specified, only clear entries for this user (requires app_name).
    """
    await self._init_db()

    async with aiosqlite.connect(self._db_path) as db:
      if app_name and user_id:
        await db.execute(
            'DELETE FROM memory_entries WHERE app_name = ? AND user_id = ?',
            (app_name, user_id),
        )
      elif app_name:
        await db.execute(
            'DELETE FROM memory_entries WHERE app_name = ?', (app_name,)
        )
      else:
        await db.execute('DELETE FROM memory_entries')

      await db.commit()
      logger.info('Cleared memory entries from database')

  async def get_memory_stats(self) -> dict:
    """Returns statistics about the memory database.

    Returns:
        Dictionary containing database statistics.
    """
    await self._init_db()

    async with aiosqlite.connect(self._db_path) as db:
      # Total entries
      async with db.execute('SELECT COUNT(*) FROM memory_entries') as cursor:
        row = await cursor.fetchone()
        total_entries = row[0] if row else 0

      # Entries per app
      entries_per_app = {}
      async with db.execute("""
          SELECT app_name, COUNT(*) 
          FROM memory_entries 
          GROUP BY app_name
      """) as cursor:
        async for app_name, count in cursor:
          entries_per_app[app_name] = count

      # Database file size
      db_size_bytes = (
          self._db_path.stat().st_size if self._db_path.exists() else 0
      )

      return {
          'total_entries': total_entries,
          'entries_per_app': entries_per_app,
          'database_file_size_bytes': db_size_bytes,
          'database_path': str(self._db_path),
      }
