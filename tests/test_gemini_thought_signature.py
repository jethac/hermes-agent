"""Integration tests for Gemini thought signature persistence.

Tests that provider_metadata (containing thought signatures) survives
session persistence and restoration, ensuring Gemini 3 conversations
work correctly across restarts.
"""

import json
import pytest
from pathlib import Path

from hermes_state import SessionDB


@pytest.fixture()
def db(tmp_path):
    """Create a SessionDB with a temp database file."""
    db_path = tmp_path / "test_gemini.db"
    session_db = SessionDB(db_path=db_path)
    yield session_db
    session_db.close()


class TestGeminiThoughtSignature:
    """Tests for Gemini thought signature roundtrip through database."""

    def test_thought_signature_roundtrip(self, db):
        """Thought signatures should survive save and restore cycle."""
        # Simulate what run_agent.py does when it gets a Gemini response
        session_id = "gemini-session-1"
        db.create_session(session_id=session_id, source="cli", model="gemini-2.5-pro-experimental")

        # User message
        db.append_message(session_id, role="user", content="What is 2+2?")

        # Assistant message with tool calls containing extra_content
        thought_signature = "thought_sig_abc123xyz"
        tool_calls = [
            {
                "id": "call_calc_1",
                "function": {"name": "calculate", "arguments": '{"expr": "2+2"}'},
                "extra_content": thought_signature,
            }
        ]
        provider_metadata = {"extra_content": thought_signature}

        db.append_message(
            session_id,
            role="assistant",
            content="",
            tool_calls=tool_calls,
            provider_metadata=provider_metadata,
        )

        # Tool response
        db.append_message(
            session_id,
            role="tool",
            content="4",
            tool_call_id="call_calc_1",
            tool_name="calculate",
        )

        # --- Session restore simulation ---
        # This is what happens when the gateway/CLI restores a session
        conversation = db.get_messages_as_conversation(session_id)

        # Verify structure
        assert len(conversation) == 3
        assert conversation[0]["role"] == "user"
        assert conversation[1]["role"] == "assistant"
        assert conversation[2]["role"] == "tool"

        # Verify extra_content was restored to tool_calls
        assert "tool_calls" in conversation[1]
        assert len(conversation[1]["tool_calls"]) == 1
        restored_tc = conversation[1]["tool_calls"][0]
        assert restored_tc.get("extra_content") == thought_signature

    def test_multiple_tool_calls_share_signature(self, db):
        """All tool calls in a single message should get the same signature."""
        session_id = "gemini-session-2"
        db.create_session(session_id=session_id, source="cli")

        thought_signature = "sig_parallel_xyz"
        tool_calls = [
            {"id": "call_1", "function": {"name": "search", "arguments": "{}"}},
            {"id": "call_2", "function": {"name": "calculate", "arguments": "{}"}},
            {"id": "call_3", "function": {"name": "read_file", "arguments": "{}"}},
        ]
        provider_metadata = {"extra_content": thought_signature}

        db.append_message(
            session_id,
            role="assistant",
            content="",
            tool_calls=tool_calls,
            provider_metadata=provider_metadata,
        )

        conversation = db.get_messages_as_conversation(session_id)
        assert len(conversation) == 1
        restored_calls = conversation[0]["tool_calls"]
        assert len(restored_calls) == 3

        # All tool calls should have the same signature
        for tc in restored_calls:
            assert tc.get("extra_content") == thought_signature

    def test_session_without_thought_signature_unaffected(self, db):
        """Non-Gemini sessions (or Gemini without tool calls) work normally."""
        session_id = "non-gemini-session"
        db.create_session(session_id=session_id, source="cli", model="gpt-4")

        db.append_message(session_id, role="user", content="Hello")
        db.append_message(session_id, role="assistant", content="Hi there!")

        conversation = db.get_messages_as_conversation(session_id)
        assert len(conversation) == 2
        assert conversation[0]["content"] == "Hello"
        assert conversation[1]["content"] == "Hi there!"

    def test_mixed_messages_with_and_without_metadata(self, db):
        """A session with both metadata and non-metadata messages works correctly."""
        session_id = "mixed-session"
        db.create_session(session_id=session_id, source="cli")

        # Regular message
        db.append_message(session_id, role="user", content="Start")
        db.append_message(session_id, role="assistant", content="OK")

        # Message with thought signature
        tool_calls = [{"id": "call_1", "function": {"name": "search", "arguments": "{}"}}]
        db.append_message(
            session_id,
            role="assistant",
            content="",
            tool_calls=tool_calls,
            provider_metadata={"extra_content": "sig_123"},
        )

        # Another regular message
        db.append_message(session_id, role="tool", content="result", tool_call_id="call_1")
        db.append_message(session_id, role="assistant", content="Done")

        conversation = db.get_messages_as_conversation(session_id)
        assert len(conversation) == 5

        # First two messages have no metadata
        assert "tool_calls" not in conversation[0]
        assert "tool_calls" not in conversation[1]

        # Third message has tool_calls with extra_content
        assert conversation[2]["tool_calls"][0]["extra_content"] == "sig_123"

        # Last two messages have no metadata
        assert "tool_calls" not in conversation[3]
        assert "tool_calls" not in conversation[4]

    def test_malformed_provider_metadata_graceful(self, db):
        """Invalid JSON in provider_metadata should not crash restoration."""
        session_id = "malformed-session"
        db.create_session(session_id=session_id, source="cli")

        # Manually insert a message with invalid JSON metadata
        db._conn.execute(
            """INSERT INTO messages (session_id, role, content, timestamp, provider_metadata)
               VALUES (?, ?, ?, ?, ?)""",
            (session_id, "user", "Hello", 1000.0, "not valid json{"),
        )
        db._conn.commit()

        # Should not crash — just skip the metadata
        conversation = db.get_messages_as_conversation(session_id)
        assert len(conversation) == 1
        assert conversation[0]["content"] == "Hello"

    def test_extra_content_not_added_to_non_tool_messages(self, db):
        """Provider metadata with extra_content should only affect tool_calls."""
        session_id = "text-only-session"
        db.create_session(session_id=session_id, source="cli")

        # Message with provider_metadata but NO tool_calls
        # (This shouldn't happen in practice, but test defensive handling)
        db.append_message(
            session_id,
            role="assistant",
            content="Just text",
            provider_metadata={"extra_content": "sig_orphan"},
        )

        conversation = db.get_messages_as_conversation(session_id)
        assert len(conversation) == 1
        # Should not have tool_calls added
        assert "tool_calls" not in conversation[0]
        assert conversation[0]["content"] == "Just text"

    def test_empty_provider_metadata(self, db):
        """Empty provider_metadata dict should be handled gracefully."""
        session_id = "empty-metadata-session"
        db.create_session(session_id=session_id, source="cli")

        tool_calls = [{"id": "call_1", "function": {"name": "test", "arguments": "{}"}}]
        db.append_message(
            session_id,
            role="assistant",
            content="",
            tool_calls=tool_calls,
            provider_metadata={},  # Empty dict
        )

        conversation = db.get_messages_as_conversation(session_id)
        assert len(conversation) == 1
        # Tool calls should exist but without extra_content
        assert "tool_calls" in conversation[0]
        assert "extra_content" not in conversation[0]["tool_calls"][0]


class TestExtraContentPreservation:
    """Tests for general provider metadata preservation beyond thought signatures."""

    def test_arbitrary_metadata_structure(self, db):
        """Provider metadata can store arbitrary JSON structures."""
        session_id = "arbitrary-session"
        db.create_session(session_id=session_id, source="cli")

        metadata = {
            "extra_content": "sig_abc",
            "custom_field": "value",
            "nested": {"key": "data"},
        }

        db.append_message(
            session_id,
            role="assistant",
            content="test",
            provider_metadata=metadata,
        )

        messages = db.get_messages(session_id)
        stored = json.loads(messages[0]["provider_metadata"])
        assert stored == metadata

    def test_provider_metadata_in_export(self, db):
        """Exported sessions should include provider_metadata."""
        session_id = "export-session"
        db.create_session(session_id=session_id, source="cli")

        db.append_message(
            session_id,
            role="assistant",
            content="test",
            provider_metadata={"extra_content": "sig_export"},
        )

        export = db.export_session(session_id)
        assert len(export["messages"]) == 1
        metadata = json.loads(export["messages"][0]["provider_metadata"])
        assert metadata["extra_content"] == "sig_export"
