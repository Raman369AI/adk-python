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

from pathlib import Path
import tempfile
from unittest.mock import AsyncMock
from unittest.mock import Mock

from google.adk.events.event import Event
from google.adk.memory.sqlite_memory_service import SqliteMemoryService
from google.adk.sessions.session import Session
from google.genai import types
import pytest


@pytest.fixture
def temp_db_path():
  """Provides a temporary database file path for testing."""
  with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as tmp:
    yield tmp.name
  # Cleanup
  Path(tmp.name).unlink(missing_ok=True)


@pytest.fixture
def sqlite_service(temp_db_path):
  """Provides a SqliteMemoryService instance with temp database."""
  return SqliteMemoryService(db_path=temp_db_path)


@pytest.fixture
def sample_session():
  """Creates a sample session with events for testing."""
  session = Mock(spec=Session)
  session.id = 'test_session_123'
  session.app_name = 'test_app'
  session.user_id = 'user_456'

  # Create test events
  event1 = Mock(spec=Event)
  event1.author = 'user'
  event1.timestamp = 1640995200.0  # 2022-01-01 00:00:00
  event1.content = types.Content(
      parts=[types.Part(text='Hello world')], role='user'
  )

  event2 = Mock(spec=Event)
  event2.author = 'assistant'
  event2.timestamp = 1640995260.0  # 2022-01-01 00:01:00
  event2.content = types.Content(
      parts=[types.Part(text='How can I help you today?')], role='model'
  )

  # Event with no text parts (should be filtered out)
  event3 = Mock(spec=Event)
  event3.author = 'system'
  event3.timestamp = 1640995320.0
  event3.content = types.Content(parts=[], role='system')

  session.events = [event1, event2, event3]
  return session


@pytest.mark.asyncio
class TestSqliteMemoryService:
  """Test suite for SqliteMemoryService."""

  async def test_init_creates_database(self, temp_db_path):
    """Test that database is created during initialization."""
    service = SqliteMemoryService(db_path=temp_db_path)
    await service._init_db()

    assert Path(temp_db_path).exists()

  async def test_add_session_to_memory(self, sqlite_service, sample_session):
    """Test adding a session to memory."""
    await sqlite_service.add_session_to_memory(sample_session)

    stats = await sqlite_service.get_memory_stats()
    assert stats['total_entries'] == 2  # Only events with text content
    assert stats['entries_per_app']['test_app'] == 2

  async def test_search_memory_keyword_matching(
      self, sqlite_service, sample_session
  ):
    """Test searching memory using keyword matching."""
    await sqlite_service.add_session_to_memory(sample_session)

    # Search for matching keyword
    response = await sqlite_service.search_memory(
        app_name='test_app', user_id='user_456', query='hello'
    )

    assert len(response.memories) == 1
    assert response.memories[0].author == 'user'
    assert 'Hello world' in response.memories[0].content.parts[0].text

  async def test_search_memory_no_matches(self, sqlite_service, sample_session):
    """Test searching memory with no matching keywords."""
    await sqlite_service.add_session_to_memory(sample_session)

    response = await sqlite_service.search_memory(
        app_name='test_app', user_id='user_456', query='nonexistent'
    )

    assert len(response.memories) == 0

  async def test_search_memory_multiple_keywords(
      self, sqlite_service, sample_session
  ):
    """Test searching memory with multiple keywords."""
    await sqlite_service.add_session_to_memory(sample_session)

    response = await sqlite_service.search_memory(
        app_name='test_app', user_id='user_456', query='help today'
    )

    assert len(response.memories) == 1
    assert response.memories[0].author == 'assistant'

  async def test_search_memory_user_isolation(self, sqlite_service):
    """Test that memory search is isolated by user."""
    # Create sessions for different users
    session1 = Mock(spec=Session)
    session1.id = 'session1'
    session1.app_name = 'test_app'
    session1.user_id = 'user1'
    session1.events = [
        Mock(
            spec=Event,
            author='user1',
            timestamp=1640995200.0,
            content=types.Content(
                parts=[types.Part(text='user1 message')], role='user'
            ),
        )
    ]

    session2 = Mock(spec=Session)
    session2.id = 'session2'
    session2.app_name = 'test_app'
    session2.user_id = 'user2'
    session2.events = [
        Mock(
            spec=Event,
            author='user2',
            timestamp=1640995200.0,
            content=types.Content(
                parts=[types.Part(text='user2 message')], role='user'
            ),
        )
    ]

    await sqlite_service.add_session_to_memory(session1)
    await sqlite_service.add_session_to_memory(session2)

    # Search for user1 should only return user1's memories
    response = await sqlite_service.search_memory(
        app_name='test_app', user_id='user1', query='message'
    )

    assert len(response.memories) == 1
    assert response.memories[0].author == 'user1'

  async def test_clear_memory_all(self, sqlite_service, sample_session):
    """Test clearing all memory entries."""
    await sqlite_service.add_session_to_memory(sample_session)

    stats_before = await sqlite_service.get_memory_stats()
    assert stats_before['total_entries'] > 0

    await sqlite_service.clear_memory()

    stats_after = await sqlite_service.get_memory_stats()
    assert stats_after['total_entries'] == 0

  async def test_clear_memory_by_app(self, sqlite_service):
    """Test clearing memory entries by app name."""
    # Create sessions for different apps
    session1 = Mock(spec=Session)
    session1.id = 'session1'
    session1.app_name = 'app1'
    session1.user_id = 'user1'
    session1.events = [
        Mock(
            spec=Event,
            author='user',
            timestamp=1640995200.0,
            content=types.Content(
                parts=[types.Part(text='app1 message')], role='user'
            ),
        )
    ]

    session2 = Mock(spec=Session)
    session2.id = 'session2'
    session2.app_name = 'app2'
    session2.user_id = 'user1'
    session2.events = [
        Mock(
            spec=Event,
            author='user',
            timestamp=1640995200.0,
            content=types.Content(
                parts=[types.Part(text='app2 message')], role='user'
            ),
        )
    ]

    await sqlite_service.add_session_to_memory(session1)
    await sqlite_service.add_session_to_memory(session2)

    await sqlite_service.clear_memory(app_name='app1')

    stats = await sqlite_service.get_memory_stats()
    assert stats['total_entries'] == 1
    assert 'app1' not in stats['entries_per_app']
    assert stats['entries_per_app']['app2'] == 1

  async def test_clear_memory_by_user(self, sqlite_service):
    """Test clearing memory entries by user within an app."""
    # Create sessions for different users in same app
    session1 = Mock(spec=Session)
    session1.id = 'session1'
    session1.app_name = 'test_app'
    session1.user_id = 'user1'
    session1.events = [
        Mock(
            spec=Event,
            author='user1',
            timestamp=1640995200.0,
            content=types.Content(
                parts=[types.Part(text='user1 message')], role='user'
            ),
        )
    ]

    session2 = Mock(spec=Session)
    session2.id = 'session2'
    session2.app_name = 'test_app'
    session2.user_id = 'user2'
    session2.events = [
        Mock(
            spec=Event,
            author='user2',
            timestamp=1640995200.0,
            content=types.Content(
                parts=[types.Part(text='user2 message')], role='user'
            ),
        )
    ]

    await sqlite_service.add_session_to_memory(session1)
    await sqlite_service.add_session_to_memory(session2)

    await sqlite_service.clear_memory(app_name='test_app', user_id='user1')

    # Verify user1's memories are gone but user2's remain
    response = await sqlite_service.search_memory(
        app_name='test_app', user_id='user1', query='message'
    )
    assert len(response.memories) == 0

    response = await sqlite_service.search_memory(
        app_name='test_app', user_id='user2', query='message'
    )
    assert len(response.memories) == 1

  async def test_get_memory_stats(self, sqlite_service, sample_session):
    """Test getting memory statistics."""
    await sqlite_service.add_session_to_memory(sample_session)

    stats = await sqlite_service.get_memory_stats()

    assert isinstance(stats, dict)
    assert 'total_entries' in stats
    assert 'entries_per_app' in stats
    assert 'database_file_size_bytes' in stats
    assert 'database_path' in stats

    assert stats['total_entries'] == 2
    assert stats['entries_per_app']['test_app'] == 2
    assert stats['database_file_size_bytes'] > 0

  async def test_duplicate_session_handling(
      self, sqlite_service, sample_session
  ):
    """Test that duplicate sessions are handled properly."""
    await sqlite_service.add_session_to_memory(sample_session)
    await sqlite_service.add_session_to_memory(sample_session)  # Add again

    stats = await sqlite_service.get_memory_stats()
    # Should still be 2 entries due to UNIQUE constraint
    assert stats['total_entries'] == 2

  async def test_empty_content_filtering(self, sqlite_service):
    """Test that events with empty content are filtered out."""
    session = Mock(spec=Session)
    session.id = 'test_session'
    session.app_name = 'test_app'
    session.user_id = 'test_user'

    # Event with no content
    event1 = Mock(spec=Event)
    event1.content = None
    event1.timestamp = 1640995200.0

    # Event with empty parts
    event2 = Mock(spec=Event)
    event2.content = types.Content(parts=[], role='user')
    event2.timestamp = 1640995260.0

    # Valid event
    event3 = Mock(spec=Event)
    event3.author = 'user'
    event3.timestamp = 1640995320.0
    event3.content = types.Content(
        parts=[types.Part(text='Valid message')], role='user'
    )

    session.events = [event1, event2, event3]

    await sqlite_service.add_session_to_memory(session)

    stats = await sqlite_service.get_memory_stats()
    assert stats['total_entries'] == 1  # Only the valid event
