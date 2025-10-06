"""Tests for bookwiki Conversation model."""

from bookwiki.db import SafeConnection
from bookwiki.models import Block, Conversation
from tests.helpers import (
    verify_all_blocks_sent,
    verify_conversation_token_state,
    verify_conversation_waiting_state,
)


def test_conversation_create(temp_db: SafeConnection) -> None:
    """Test creating a new conversation."""
    with temp_db.transaction_cursor() as cursor:
        # Create a conversation without parent
        conversation = Conversation.create(cursor)

        assert conversation.id is not None
        assert conversation.previously is None
        assert conversation.parent_block_id is None
        assert conversation.total_input_tokens == 0
        assert conversation.total_output_tokens == 0
        assert conversation.current_tokens == 0


def test_conversation_create_with_parent(temp_db: SafeConnection) -> None:
    """Test creating a conversation with a parent block."""
    with temp_db.transaction_cursor() as cursor:
        # Create parent conversation and block
        parent_conversation = Conversation.create(cursor)
        parent_block = Block.create_text(
            cursor=cursor,
            conversation_id=parent_conversation.id,
            generation=parent_conversation.current_generation,
            role="assistant",
            text="Starting a sub-agent",
        )

        # Create child conversation
        child_conversation = Conversation.create(cursor, parent_block.id)

        assert child_conversation.id != parent_conversation.id
        assert child_conversation.parent_block_id == parent_block.id
        assert child_conversation.previously is None


def test_conversation_add_blocks(temp_db: SafeConnection) -> None:
    """Test adding different types of blocks to a conversation."""
    with temp_db.transaction_cursor() as cursor:
        conversation = Conversation.create(cursor)

        # Add user text
        user_block = conversation.add_user_text("Hello, can you help me?")
        assert user_block.text_role == "user"
        assert user_block.text_body == "Hello, can you help me?"
        assert user_block.sent is False
        assert user_block.conversation_id == conversation.id

        # Add assistant text
        assistant_block = conversation.add_assistant_text("Sure, I'd be happy to help!")
        assert assistant_block.text_role == "assistant"
        assert assistant_block.text_body == "Sure, I'd be happy to help!"
        assert assistant_block.sent is True  # Assistant messages are marked as sent
        assert assistant_block.conversation_id == conversation.id

        # Add tool use
        tool_block = conversation.add_tool_use(
            name="ReadChapter", use_id="tool_123", params='{"chapter_offset": 0}'
        )
        assert tool_block.tool_name == "ReadChapter"
        assert tool_block.tool_use_id == "tool_123"
        assert tool_block.tool_params == '{"chapter_offset": 0}'
        assert tool_block.conversation_id == conversation.id


def test_conversation_blocks(temp_db: SafeConnection) -> None:
    """Test retrieving all blocks for a conversation in order."""
    with temp_db.transaction_cursor() as cursor:
        conversation = Conversation.create(cursor)

        # Add blocks in a specific order
        user_block = conversation.add_user_text("First message")
        tool_block = conversation.add_tool_use(
            name="TestTool", use_id="tool_1", params="{}"
        )
        assistant_block = conversation.add_assistant_text("Response message")

        # Get all blocks
        all_blocks = conversation.blocks

        # Should be in order of creation (by id)
        assert len(all_blocks) == 3
        assert all_blocks[0].id == user_block.id
        assert all_blocks[1].id == tool_block.id
        assert all_blocks[2].id == assistant_block.id

        # Verify content
        assert all_blocks[0].text_body == "First message"
        assert all_blocks[1].tool_name == "TestTool"
        assert all_blocks[2].text_body == "Response message"


def test_conversation_update_tokens(temp_db: SafeConnection) -> None:
    """Test updating token counts for a conversation."""
    with temp_db.transaction_cursor() as cursor:
        conversation = Conversation.create(cursor)

        # Initial token counts should be zero
        assert conversation.total_input_tokens == 0
        assert conversation.total_output_tokens == 0
        assert conversation.current_tokens == 0

        # Update tokens
        conversation.update_tokens(100, 50)

        # Verify the database was updated using helper
        verify_conversation_token_state(cursor, conversation, 100, 50, 150)

        # Update tokens again (should accumulate)
        conversation.update_tokens(25, 35)

        # Verify accumulated totals using helper
        verify_conversation_token_state(
            cursor, conversation, 125, 85, 60
        )  # 25 + 35 (latest only)


def test_conversation_find_sendable_conversation_empty(temp_db: SafeConnection) -> None:
    """Test finding sendable conversation when none exist."""
    with temp_db.transaction_cursor() as cursor:
        # No conversations exist
        ready_conversation = Conversation.find_sendable_conversation(cursor)
        assert ready_conversation is None

        # Create a conversation with no blocks
        Conversation.create(cursor)
        ready_conversation = Conversation.find_sendable_conversation(cursor)
        assert ready_conversation is None  # No unsent blocks


def test_conversation_find_sendable_conversation_with_unsent_text(
    temp_db: SafeConnection,
) -> None:
    """Test finding conversation with unsent text blocks."""
    with temp_db.transaction_cursor() as cursor:
        conversation = Conversation.create(cursor)

        # Add an unsent user message
        conversation.add_user_text("Hello")

        # Should be ready (has unsent blocks, no incomplete tool uses)
        ready_conversation = Conversation.find_sendable_conversation(cursor)
        assert ready_conversation is not None
        assert ready_conversation.id == conversation.id


def test_conversation_find_sendable_conversation_with_incomplete_tools(
    temp_db: SafeConnection,
) -> None:
    """Test that conversations with incomplete tool uses are not ready."""
    with temp_db.transaction_cursor() as cursor:
        conversation = Conversation.create(cursor)

        # Add an unsent user message
        conversation.add_user_text("Please read chapter 1")

        # Add a tool use without a response
        tool_block = conversation.add_tool_use(
            name="ReadChapter", use_id="tool_1", params='{"chapter_offset": 0}'
        )

        # Should NOT be ready (has incomplete tool use)
        ready_conversation = Conversation.find_sendable_conversation(cursor)
        assert ready_conversation is None

        # Complete the tool use
        tool_block.respond("Chapter content here...")

        # Now should be ready
        ready_conversation = Conversation.find_sendable_conversation(cursor)
        assert ready_conversation is not None
        assert ready_conversation.id == conversation.id


def test_conversation_find_sendable_conversation_priority(
    temp_db: SafeConnection,
) -> None:
    """Test finding sendable conversation returns earliest by id."""
    with temp_db.transaction_cursor() as cursor:
        # Create first conversation - ready
        conv1 = Conversation.create(cursor)
        conv1.add_user_text("First conversation message")

        # Create second conversation - not ready (incomplete tool)
        conv2 = Conversation.create(cursor)
        conv2.add_user_text("Second conversation message")
        conv2.add_tool_use("TestTool", "tool_1", "{}")

        # Create third conversation - ready
        conv3 = Conversation.create(cursor)
        conv3.add_user_text("Third conversation message")

        # Create fourth conversation - no unsent blocks
        conv4 = Conversation.create(cursor)
        sent_block = conv4.add_user_text("Already sent message")
        sent_block.mark_as_sent()

        ready_conversation = Conversation.find_sendable_conversation(cursor)

        # Should find conv1 (earliest ready conversation)
        assert ready_conversation is not None
        assert ready_conversation.id == conv1.id


def test_conversation_find_sendable_conversation_with_error_responses(
    temp_db: SafeConnection,
) -> None:
    """Test that conversations with error tool responses are still considered ready."""
    with temp_db.transaction_cursor() as cursor:
        conversation = Conversation.create(cursor)

        # Add user message and tool use
        conversation.add_user_text("Read non-existent chapter")
        tool_block = conversation.add_tool_use(
            name="ReadChapter", use_id="tool_error", params='{"chapter_offset": 999}'
        )

        # Initially not ready
        ready_conversation = Conversation.find_sendable_conversation(cursor)
        assert ready_conversation is None

        # Respond with error
        tool_block.respond_error("Chapter 999 does not exist")

        # Should now be ready (even with error response)
        ready_conversation = Conversation.find_sendable_conversation(cursor)
        assert ready_conversation is not None
        assert ready_conversation.id == conversation.id


def test_conversation_complex_interaction_pattern(temp_db: SafeConnection) -> None:
    """Test a complex conversation pattern with multiple interactions."""
    with temp_db.transaction_cursor() as cursor:
        conversation = Conversation.create(cursor)

        # User starts conversation
        user_text = conversation.add_user_text(
            "Please read chapter 1 and create a wiki page"
        )

        # Should be ready
        ready = Conversation.find_sendable_conversation(cursor)
        assert ready is not None
        assert ready.id == conversation.id

        # Assistant responds with tool uses
        conversation.add_assistant_text(
            "I'll read chapter 1 and create a wiki page for you."
        )
        tool1 = conversation.add_tool_use(
            "ReadChapter", "read_1", '{"chapter_offset": 0}'
        )
        tool2 = conversation.add_tool_use(
            "WriteWikiPage",
            "write_1",
            '{"slug": "test", "create": true, "title": "Test Page"}',
        )
        # Mark user text as sent (mimics real behavior where it would be sent to LLM)
        user_text.mark_as_sent()

        # Should NOT be ready (incomplete tools)
        ready = Conversation.find_sendable_conversation(cursor)
        assert ready is None

        # Complete first tool
        tool1.respond("Chapter 1 content here...")

        # Still not ready (second tool incomplete)
        ready = Conversation.find_sendable_conversation(cursor)
        assert ready is None

        # Complete second tool
        tool2.respond("Wiki page created successfully")

        # Now should be ready again
        ready = Conversation.find_sendable_conversation(cursor)
        assert ready is not None
        assert ready.id == conversation.id

        # Verify all blocks are present
        all_blocks = conversation.blocks
        assert len(all_blocks) == 4

        # Check the sequence
        assert all_blocks[0].text_role == "user"
        assert all_blocks[1].text_role == "assistant"
        assert all_blocks[2].tool_name == "ReadChapter"
        assert all_blocks[3].tool_name == "WriteWikiPage"
        assert all_blocks[3].id == tool2.id  # Last block should be the second tool use
        assert all_blocks[3].tool_response == "Wiki page created successfully"


def test_conversation_create_with_null_waiting_on_id(temp_db: SafeConnection) -> None:
    """Test that new conversations are created with waiting_on_id as None."""
    with temp_db.transaction_cursor() as cursor:
        conv = Conversation.create(cursor)

        assert conv.waiting_on_id is None

        # Verify in database using helper
        verify_conversation_waiting_state(cursor, conv, None)


def test_find_sendable_conversation_excludes_waiting(temp_db: SafeConnection) -> None:
    """Test find_sendable_conversation excludes conversations with waiting_on_id."""
    with temp_db.transaction_cursor() as cursor:
        # Create a conversation that's ready (has unsent blocks)
        conv1 = Conversation.create(cursor)
        conv1.add_user_text("Test message 1")

        # Create another conversation that's ready but waiting on something
        conv2 = Conversation.create(cursor)
        conv2.add_user_text("Test message 2")
        conv2.set_waiting_on_id("some_batch_id")

        # Create a conversation that's not ready (no unsent blocks)
        Conversation.create(cursor)

        # Only conv1 should be returned
        ready = Conversation.find_sendable_conversation(cursor)
        assert ready is not None
        assert ready.id == conv1.id


def test_find_waiting_conversation(temp_db: SafeConnection) -> None:
    """Test find_waiting_conversation returns earliest waiting conversation."""
    with temp_db.transaction_cursor() as cursor:
        # Create conversations with different states
        Conversation.create(cursor)  # Not waiting

        conv2 = Conversation.create(cursor)
        conv2.set_waiting_on_id("batch_123")

        conv3 = Conversation.create(cursor)
        conv3.set_waiting_on_id("batch_456")

        # Should return conv2 (earliest waiting conversation)
        waiting = Conversation.find_waiting_conversation(cursor)
        assert waiting is not None
        assert waiting.id == conv2.id
        assert waiting.waiting_on_id == "batch_123"


def test_conversation_waiting_on_id_preserved_in_queries(
    temp_db: SafeConnection,
) -> None:
    """Test that waiting_on_id is preserved when loading conversations from database."""
    with temp_db.transaction_cursor() as cursor:
        # Create a conversation and set waiting_on_id
        conv = Conversation.create(cursor)
        conv.add_user_text("Test")
        test_id = "test_waiting_id_789"
        conv.set_waiting_on_id(test_id)

        # Load via find_waiting_conversation
        waiting = Conversation.find_waiting_conversation(cursor)
        assert waiting is not None
        assert waiting.waiting_on_id == test_id

        # Verify it's excluded from find_sendable_conversation
        ready = Conversation.find_sendable_conversation(cursor)
        assert ready is None


def test_conversation_mark_all_blocks_as_sent(temp_db: SafeConnection) -> None:
    """Test marking all unsent blocks in a conversation as sent."""
    with temp_db.transaction_cursor() as cursor:
        conv = Conversation.create(cursor)

        # Add multiple blocks, some already sent
        conv.add_user_text("First message")
        conv.add_assistant_text("Response")
        Block.create_text(
            cursor, conv.id, conv.current_generation, "user", "Already sent", sent=True
        )
        conv.add_tool_use("TestTool", "tool-123", '{"param": "value"}')

        # Mark all blocks as sent
        conv.mark_all_blocks_as_sent()

        # Verify all blocks are now marked as sent using helper
        verify_all_blocks_sent(cursor, conv)


def test_all_conversations_finished(temp_db: SafeConnection) -> None:
    """Test all_conversations_finished method in various scenarios."""
    with temp_db.transaction_cursor() as cursor:
        # Initially, with no conversations, all should be finished
        assert Conversation.all_conversations_finished(cursor) is True

        # Create a conversation with unsent blocks
        conv1 = Conversation.create(cursor)
        conv1.add_user_text("Hello")
        assert Conversation.all_conversations_finished(cursor) is False

        # Mark blocks as sent - should still be unfinished if we have other active convs
        conv1.mark_all_blocks_as_sent()
        assert Conversation.all_conversations_finished(cursor) is True

        # Create another conversation waiting on LLM response
        conv2 = Conversation.create(cursor)
        conv2.add_user_text("Question")
        conv2.mark_all_blocks_as_sent()
        conv2.set_waiting_on_id("batch_123")
        assert Conversation.all_conversations_finished(cursor) is False

        # Clear waiting_on_id - should be finished now
        conv2.set_waiting_on_id(None)
        assert Conversation.all_conversations_finished(cursor) is True

        # Create conversation with both unsent blocks AND waiting_on_id
        conv3 = Conversation.create(cursor)
        conv3.add_user_text("Another question")
        conv3.set_waiting_on_id("batch_456")
        assert Conversation.all_conversations_finished(cursor) is False

        # Clear waiting but still have unsent blocks
        conv3.set_waiting_on_id(None)
        assert Conversation.all_conversations_finished(cursor) is False

        # Mark all blocks as sent
        conv3.mark_all_blocks_as_sent()
        assert Conversation.all_conversations_finished(cursor) is True

        # Test with multiple active conversations
        conv4 = Conversation.create(cursor)
        conv4.add_user_text("Conv 4")
        conv5 = Conversation.create(cursor)
        conv5.add_user_text("Conv 5")
        conv5.mark_all_blocks_as_sent()
        conv5.set_waiting_on_id("batch_789")

        # Should be false because we have both unsent blocks and waiting conversations
        assert Conversation.all_conversations_finished(cursor) is False

        # Clean up conv4
        conv4.mark_all_blocks_as_sent()
        assert Conversation.all_conversations_finished(cursor) is False

        # Clean up conv5
        conv5.set_waiting_on_id(None)
        assert Conversation.all_conversations_finished(cursor) is True


def test_get_all_conversations_stats(temp_db: SafeConnection) -> None:
    """Test getting aggregated token statistics across all conversations."""
    with temp_db.transaction_cursor() as cursor:
        # Empty database should return zero stats
        stats = Conversation.get_all_conversations_stats(cursor)
        assert stats.total_input_tokens == 0
        assert stats.total_output_tokens == 0

        # Create conversations with different token counts
        conv1 = Conversation.create(cursor)
        conv1.update_tokens(100, 200)

        conv2 = Conversation.create(cursor)
        conv2.update_tokens(50, 75)

        conv3 = Conversation.create(cursor)
        conv3.update_tokens(25, 125)

        # Test aggregation
        stats = Conversation.get_all_conversations_stats(cursor)
        assert stats.total_input_tokens == 175  # 100 + 50 + 25
        assert stats.total_output_tokens == 400  # 200 + 75 + 125

        # Add more tokens to existing conversation
        conv1.update_tokens(10, 15)
        stats = Conversation.get_all_conversations_stats(cursor)
        assert stats.total_input_tokens == 185  # 175 + 10
        assert stats.total_output_tokens == 415  # 400 + 15

        # Create conversation with zero tokens (shouldn't affect totals)
        Conversation.create(cursor)
        stats = Conversation.get_all_conversations_stats(cursor)
        assert stats.total_input_tokens == 185
        assert stats.total_output_tokens == 415


def test_conversation_detect_serial_tool_use_no_previous_generation(
    temp_db: SafeConnection,
) -> None:
    """Test detect_serial_tool_use with no previous generation."""
    with temp_db.transaction_cursor() as cursor:
        conversation = Conversation.create(cursor)

        # Add a tool use in generation 0
        conversation.add_tool_use("ReadChapter", "tool1", '{"chapter_offset": 0}')

        # Should return False - no previous generation to compare
        assert conversation.detect_serial_tool_use() is False


def test_conversation_detect_serial_tool_use_true_case(temp_db: SafeConnection) -> None:
    """Test detect_serial_tool_use returns True for serial tool usage."""
    with temp_db.transaction_cursor() as cursor:
        conversation = Conversation.create(cursor)

        # Generation 0: Add one tool use
        tool_block1 = conversation.add_tool_use(
            "ReadChapter", "tool1", '{"chapter_offset": 0}'
        )
        tool_block1.respond("Chapter content here")

        # Increment to generation 1
        conversation = conversation.increment_generation()

        # Generation 1: Add the same tool type (one use)
        tool_block2 = conversation.add_tool_use(
            "ReadChapter", "tool2", '{"chapter_offset": 0}'
        )
        tool_block2.respond("More chapter content")

        # Should detect serial usage
        assert conversation.detect_serial_tool_use() is True


def test_conversation_detect_serial_tool_use_different_tools(
    temp_db: SafeConnection,
) -> None:
    """Test detect_serial_tool_use with different tool types."""
    with temp_db.transaction_cursor() as cursor:
        conversation = Conversation.create(cursor)

        # Generation 0: ReadChapter tool
        tool_block1 = conversation.add_tool_use(
            "ReadChapter", "tool1", '{"chapter_offset": 0}'
        )
        tool_block1.respond("Chapter content")

        # Increment to generation 1
        conversation = conversation.increment_generation()

        # Generation 1: Different tool type
        tool_block2 = conversation.add_tool_use(
            "WriteWikiPage", "tool2", '{"title": "Test"}'
        )
        tool_block2.respond("Wiki page created")

        # Should not detect serial usage (different tool types)
        assert conversation.detect_serial_tool_use() is False


def test_conversation_detect_serial_tool_use_multiple_tools_current(
    temp_db: SafeConnection,
) -> None:
    """Test detect_serial_tool_use with multiple tools in current generation."""
    with temp_db.transaction_cursor() as cursor:
        conversation = Conversation.create(cursor)

        # Generation 0: One ReadChapter tool
        tool_block1 = conversation.add_tool_use(
            "ReadChapter", "tool1", '{"chapter_offset": 0}'
        )
        tool_block1.respond("Chapter content")

        # Increment to generation 1
        conversation = conversation.increment_generation()

        # Generation 1: Multiple tools (two ReadChapter)
        tool_block2 = conversation.add_tool_use(
            "ReadChapter", "tool2", '{"chapter_offset": 0}'
        )
        tool_block2.respond("More content")
        tool_block3 = conversation.add_tool_use(
            "ReadChapter", "tool3", '{"chapter_offset": 0}'
        )
        tool_block3.respond("Even more content")

        # Should not detect serial usage (multiple tools in current generation)
        assert conversation.detect_serial_tool_use() is False


def test_conversation_detect_serial_tool_use_multiple_tools_previous(
    temp_db: SafeConnection,
) -> None:
    """Test detect_serial_tool_use with multiple tools in previous generation."""
    with temp_db.transaction_cursor() as cursor:
        conversation = Conversation.create(cursor)

        # Generation 0: Multiple ReadChapter tools
        tool_block1 = conversation.add_tool_use(
            "ReadChapter", "tool1", '{"chapter_offset": 0}'
        )
        tool_block1.respond("Chapter 1")
        tool_block2 = conversation.add_tool_use(
            "ReadChapter", "tool2", '{"chapter_offset": 0}'
        )
        tool_block2.respond("Chapter 2")

        # Increment to generation 1
        conversation = conversation.increment_generation()

        # Generation 1: One ReadChapter tool
        tool_block3 = conversation.add_tool_use(
            "ReadChapter", "tool3", '{"chapter_offset": 0}'
        )
        tool_block3.respond("Chapter 3")

        # Should not detect serial usage (multiple tools in previous generation)
        assert conversation.detect_serial_tool_use() is False


def test_conversation_detect_serial_tool_use_mixed_tool_types(
    temp_db: SafeConnection,
) -> None:
    """Test detect_serial_tool_use with mixed tool types in generations."""
    with temp_db.transaction_cursor() as cursor:
        conversation = Conversation.create(cursor)

        # Generation 0: ReadChapter and WriteWikiPage
        tool_block1 = conversation.add_tool_use(
            "ReadChapter", "tool1", '{"chapter_offset": 0}'
        )
        tool_block1.respond("Chapter content")
        tool_block2 = conversation.add_tool_use(
            "WriteWikiPage", "tool2", '{"title": "Test"}'
        )
        tool_block2.respond("Wiki created")

        # Increment to generation 1
        conversation = conversation.increment_generation()

        # Generation 1: One ReadChapter tool
        tool_block3 = conversation.add_tool_use(
            "ReadChapter", "tool3", '{"chapter_offset": 0}'
        )
        tool_block3.respond("More content")

        # Should not detect serial usage (previous generation has multiple tools)
        # We only suggest parallelization when there's exactly one tool per generation
        assert conversation.detect_serial_tool_use() is False


def test_conversation_detect_serial_tool_use_no_tools_in_generation(
    temp_db: SafeConnection,
) -> None:
    """Test detect_serial_tool_use with no tools in one generation."""
    with temp_db.transaction_cursor() as cursor:
        conversation = Conversation.create(cursor)

        # Generation 0: Only text, no tools
        conversation.add_assistant_text("Just some text")

        # Increment to generation 1
        conversation = conversation.increment_generation()

        # Generation 1: One tool
        tool_block = conversation.add_tool_use(
            "ReadChapter", "tool1", '{"chapter_offset": 0}'
        )
        tool_block.respond("Chapter content")

        # Should not detect serial usage (no tools in previous generation)
        assert conversation.detect_serial_tool_use() is False


def test_conversation_detect_serial_tool_use_three_generations(
    temp_db: SafeConnection,
) -> None:
    """Test detect_serial_tool_use only looks at last two generations."""
    with temp_db.transaction_cursor() as cursor:
        conversation = Conversation.create(cursor)

        # Generation 0: WriteWikiPage tool
        tool_block1 = conversation.add_tool_use(
            "WriteWikiPage", "tool1", '{"title": "Test1"}'
        )
        tool_block1.respond("Wiki 1 created")

        # Generation 1: ReadChapter tool
        conversation = conversation.increment_generation()
        tool_block2 = conversation.add_tool_use(
            "ReadChapter", "tool2", '{"chapter_offset": 0}'
        )
        tool_block2.respond("Chapter content")

        # Generation 2: ReadChapter tool again
        conversation = conversation.increment_generation()
        tool_block3 = conversation.add_tool_use(
            "ReadChapter", "tool3", '{"chapter_offset": 0}'
        )
        tool_block3.respond("More content")

        # Should detect serial usage (ReadChapter in generation 1 and 2)
        # even though generation 0 had a different tool
        assert conversation.detect_serial_tool_use() is True
