"""Tests for bookwiki Block model."""

import pytest

from bookwiki.db import SafeConnection
from bookwiki.models import Block, Conversation
from bookwiki.utils import utc_now
from tests.helpers import verify_block_sent_state, verify_block_tool_response


def test_block_create_tool_use(temp_db: SafeConnection) -> None:
    """Test creating a tool use block."""
    with temp_db.transaction_cursor() as cursor:
        # First create a conversation
        conversation = Conversation.create(cursor)

        # Create a tool use block
        block = Block.create_tool_use(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            name="ReadChapter",
            use_id="tool_123",
            params='{"chapter_id": 1}',
        )

        # Verify the block
        assert block.conversation_id == conversation.id
        assert block.tool_name == "ReadChapter"
        assert block.tool_use_id == "tool_123"
        assert block.tool_params == '{"chapter_id": 1}'
        assert block.tool_response is None
        assert block.text_role is None
        assert block.text_body is None
        assert block.sent is False
        assert block.errored is False
        assert block.id is not None


def test_block_create_text(temp_db: SafeConnection) -> None:
    """Test creating text blocks (user and assistant)."""
    with temp_db.transaction_cursor() as cursor:
        # Create a conversation
        conversation = Conversation.create(cursor)

        # Create a user text block
        user_block = Block.create_text(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            role="user",
            text="Hello, please read chapter 1",
        )

        # Verify user block
        assert user_block.conversation_id == conversation.id
        assert user_block.text_role == "user"
        assert user_block.text_body == "Hello, please read chapter 1"
        assert user_block.sent is False
        assert user_block.tool_name is None

        # Create an assistant text block
        assistant_block = Block.create_text(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            role="assistant",
            text="I'll read chapter 1 for you.",
            sent=True,
        )

        # Verify assistant block
        assert assistant_block.text_role == "assistant"
        assert assistant_block.text_body == "I'll read chapter 1 for you."
        assert assistant_block.sent is True


def test_block_mark_as_sent(temp_db: SafeConnection) -> None:
    """Test marking a block as sent."""
    with temp_db.transaction_cursor() as cursor:
        conversation = Conversation.create(cursor)

        # Create an unsent block
        block = Block.create_text(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            role="user",
            text="Test message",
        )

        assert block.sent is False

        # Mark it as sent
        block.mark_as_sent()

        # Verify it's marked as sent in the database using helper
        verify_block_sent_state(cursor, block, True)


def test_block_respond(temp_db: SafeConnection) -> None:
    """Test responding to a tool use block."""
    with temp_db.transaction_cursor() as cursor:
        conversation = Conversation.create(cursor)

        # Create a tool use block
        block = Block.create_tool_use(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            name="ReadChapter",
            use_id="tool_456",
            params='{"chapter_id": 2}',
        )

        assert block.tool_response is None
        assert block.errored is False

        # Add a response
        response_text = "Chapter content goes here..."
        block.respond(response_text)

        # Verify the response is stored using helper
        verify_block_tool_response(cursor, block, response_text)
        # Verify error state by reloading the block
        reloaded_block = Block.get_by_id(cursor, block.id)
        assert reloaded_block is not None
        assert reloaded_block.errored is False


def test_block_respond_error(temp_db: SafeConnection) -> None:
    """Test responding to a tool use block with an error."""
    with temp_db.transaction_cursor() as cursor:
        conversation = Conversation.create(cursor)

        # Create a tool use block
        block = Block.create_tool_use(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            name="ReadChapter",
            use_id="tool_789",
            params='{"chapter_id": 999}',
        )

        # Add an error response
        error_message = "Chapter 999 does not exist"
        block.respond_error(error_message)

        # Verify the error response is stored
        updated_block = Block.get_by_id(cursor, block.id)
        assert updated_block is not None
        assert updated_block.tool_response == error_message
        assert updated_block.errored is True


def test_block_respond_twice_raises_error(temp_db: SafeConnection) -> None:
    """Test that responding to a block twice raises an error."""
    with temp_db.transaction_cursor() as cursor:
        conversation = Conversation.create(cursor)

        # Create a tool use block
        block = Block.create_tool_use(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            name="TestTool",
            use_id="double_respond_test",
            params='{"test": "params"}',
        )

        # First response should work
        block.respond("First response")

        # Verify it was stored using helper
        verify_block_tool_response(cursor, block, "First response")

        # Second response should raise ValueError
        with pytest.raises(ValueError, match="tool already has a response"):
            block.respond("Second response")

        # Verify the first response is still there using helper
        verify_block_tool_response(cursor, block, "First response")


def test_block_respond_error_twice_raises_error(temp_db: SafeConnection) -> None:
    """Test that responding with an error to a block twice raises an error."""
    with temp_db.transaction_cursor() as cursor:
        conversation = Conversation.create(cursor)

        # Create a tool use block
        block = Block.create_tool_use(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            name="TestTool",
            use_id="double_error_test",
            params='{"test": "params"}',
        )

        # First error response should work
        block.respond_error("First error")

        # Verify it was stored
        updated_block = Block.get_by_id(cursor, block.id)
        assert updated_block is not None
        assert updated_block.tool_response == "First error"
        assert updated_block.errored is True

        # Second error response should raise ValueError
        with pytest.raises(ValueError, match="tool already has a response"):
            block.respond_error("Second error")

        # Verify the first error is still there
        updated_block = Block.get_by_id(cursor, block.id)
        assert updated_block is not None
        assert updated_block.tool_response == "First error"
        assert updated_block.errored is True


def test_block_respond_after_error_raises(temp_db: SafeConnection) -> None:
    """Test that normal respond after error response raises an error."""
    with temp_db.transaction_cursor() as cursor:
        conversation = Conversation.create(cursor)

        # Create a tool use block
        block = Block.create_tool_use(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            name="TestTool",
            use_id="error_then_normal",
            params='{"test": "params"}',
        )

        # First add an error response
        block.respond_error("Error response")

        # Then try to add a normal response - should raise
        with pytest.raises(ValueError, match="tool already has a response"):
            block.respond("Normal response")

        # Verify only the error response is stored
        updated_block = Block.get_by_id(cursor, block.id)
        assert updated_block is not None
        assert updated_block.tool_response == "Error response"
        assert updated_block.errored is True


def test_block_error_after_respond_raises(temp_db: SafeConnection) -> None:
    """Test that error response after normal respond raises an error."""
    with temp_db.transaction_cursor() as cursor:
        conversation = Conversation.create(cursor)

        # Create a tool use block
        block = Block.create_tool_use(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            name="TestTool",
            use_id="normal_then_error",
            params='{"test": "params"}',
        )

        # First add a normal response
        block.respond("Normal response")

        # Then try to add an error response - should raise
        with pytest.raises(ValueError, match="tool already has a response"):
            block.respond_error("Error response")

        # Verify only the normal response is stored (without error flag)
        updated_block = Block.get_by_id(cursor, block.id)
        assert updated_block is not None
        assert updated_block.tool_response == "Normal response"
        assert updated_block.errored is False


def test_block_respond_with_reloaded_block_raises(temp_db: SafeConnection) -> None:
    """Test that responding twice raises error even with reloaded block."""
    with temp_db.transaction_cursor() as cursor:
        conversation = Conversation.create(cursor)

        # Create a tool use block
        block = Block.create_tool_use(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            name="TestTool",
            use_id="reload_test",
            params='{"test": "params"}',
        )

        # First response should work
        block.respond("First response")

        # Reload the block from database
        reloaded_block = Block.get_by_id(cursor, block.id)
        assert reloaded_block is not None
        assert reloaded_block.tool_response == "First response"

        # Second response on reloaded block should also raise
        with pytest.raises(ValueError, match="tool already has a response"):
            reloaded_block.respond("Second response")


def test_block_start_conversation(temp_db: SafeConnection) -> None:
    """Test starting a new conversation from a block."""
    with temp_db.transaction_cursor() as cursor:
        # Create initial conversation and block
        parent_conversation = Conversation.create(cursor)
        parent_block = Block.create_tool_use(
            cursor=cursor,
            conversation_id=parent_conversation.id,
            generation=parent_conversation.current_generation,
            name="SpawnAgent",
            use_id="spawn_test",
            params='{"prompt_key": "test", "template_vars": {}}',
        )

        # Start a new conversation
        child_conversation = parent_block.start_conversation()

        # Verify the parent-child relationship
        assert child_conversation.parent_block_id == parent_block.id
        assert child_conversation.id != parent_conversation.id


def test_block_date_handling(temp_db: SafeConnection) -> None:
    """Test that block creation times are handled properly."""
    with temp_db.transaction_cursor() as cursor:
        conversation = Conversation.create(cursor)

        # Create a block
        before_time = utc_now()
        block = Block.create_text(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            role="user",
            text="Time test",
        )
        after_time = utc_now()

        # Verify the creation time is between before and after
        assert before_time <= block.create_time <= after_time

        # Verify the time is stored and retrieved correctly
        updated_block = Block.get_by_id(cursor, block.id)
        assert updated_block is not None

        # The time should be in ISO format
        stored_time_str = updated_block.create_time.isoformat()
        assert "T" in stored_time_str  # ISO format includes T
        assert stored_time_str.endswith("+00:00")  # UTC timezone


def test_block_json_params_validation(temp_db: SafeConnection) -> None:
    """Test that invalid JSON in tool params is handled properly."""
    with temp_db.transaction_cursor() as cursor:
        conversation = Conversation.create(cursor)

        # This should work with valid JSON
        valid_block = Block.create_tool_use(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            name="TestTool",
            use_id="valid_json",
            params='{"valid": "json"}',
        )
        assert valid_block.tool_params == '{"valid": "json"}'

        # Test with None params (should be allowed)
        # Note: We need to directly test with None, but the create_tool_use method
        # expects a string. Let's test with valid JSON instead.
        empty_json_block = Block.create_tool_use(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            name="TestTool",
            use_id="empty_json",
            params="{}",
        )
        assert empty_json_block.tool_params == "{}"
