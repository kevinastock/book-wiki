"""Conversation state machine tests for bookwiki."""

from bookwiki.db import SafeConnection
from bookwiki.models import Chapter, Conversation


def test_conversation_ready_state_complex_scenarios(temp_db: SafeConnection) -> None:
    """Test complex scenarios for conversation readiness detection."""
    with temp_db.transaction_cursor() as cursor:
        Chapter.add_chapter(cursor, 1, ["Book"], "Chapter content")

        # Scenario 1: Conversation with mixed sent/unsent blocks
        conv1 = Conversation.create(cursor)

        # Add user message (unsent by default)
        user_msg = conv1.add_user_text("Please help me")
        assert user_msg.sent is False

        # Add assistant response (sent by default)
        assistant_msg = conv1.add_assistant_text("I'll help you")
        assert assistant_msg.sent is True

        # Should be ready (has unsent user message, no incomplete tools)
        ready = Conversation.find_sendable_conversation(cursor)
        assert ready is not None
        assert ready.id == conv1.id

        # Mark user message as sent
        user_msg.mark_as_sent()

        # Should no longer be ready (no unsent blocks)
        ready = Conversation.find_sendable_conversation(cursor)
        assert ready is None

        # Scenario 2: Conversation with completed and incomplete tools
        conv2 = Conversation.create(cursor)
        conv2.add_user_text("Read chapters 1 and 2")

        # Add two tool uses
        tool1 = conv2.add_tool_use("ReadChapter", "read1", '{"chapter_offset": 0}')
        tool2 = conv2.add_tool_use("ReadChapter", "read2", '{"chapter_offset": 0}')

        # Not ready (incomplete tools)
        ready = Conversation.find_sendable_conversation(cursor)
        assert ready is None

        # Complete first tool
        tool1.respond("Chapter 1 content")

        # Still not ready (second tool incomplete)
        ready = Conversation.find_sendable_conversation(cursor)
        assert ready is None

        # Complete second tool
        tool2.respond("Chapter 2 content")

        # Now ready
        ready = Conversation.find_sendable_conversation(cursor)
        assert ready is not None
        assert ready.id == conv2.id


def test_mixed_ready_and_not_ready_conversations(temp_db: SafeConnection) -> None:
    """Test scenarios with multiple conversations in different states."""
    with temp_db.transaction_cursor() as cursor:
        Chapter.add_chapter(cursor, 1, ["Book"], "Chapter content")

        conversations = []

        # Create conversation 1: Ready (user message only)
        conv1 = Conversation.create(cursor)
        conv1.add_user_text("Simple request")
        conversations.append(("ready_simple", conv1))

        # Create conversation 2: Not ready (incomplete tool)
        conv2 = Conversation.create(cursor)
        conv2.add_user_text("Tool request")
        conv2.add_tool_use("ReadChapter", "read1", '{"chapter_offset": 0}')
        conversations.append(("not_ready_tool", conv2))

        # Create conversation 3: Not ready (no unsent blocks)
        conv3 = Conversation.create(cursor)
        sent_msg = conv3.add_user_text("Already processed")
        sent_msg.mark_as_sent()
        conv3.add_assistant_text("Response given")
        conversations.append(("not_ready_sent", conv3))

        # Create conversation 4: Ready (complex but complete)
        conv4 = Conversation.create(cursor)
        conv4.add_user_text("Complex request")
        conv4.add_assistant_text("Working on it")
        tool = conv4.add_tool_use("ReadChapter", "read2", '{"chapter_offset": 0}')
        tool.respond("Tool completed")
        conversations.append(("ready_complex", conv4))

        # Create conversation 5: Not ready (mixed complete/incomplete tools)
        conv5 = Conversation.create(cursor)
        conv5.add_user_text("Multiple tools")
        tool_a = conv5.add_tool_use("ReadChapter", "read3", '{"chapter_offset": 0}')
        conv5.add_tool_use("ReadChapter", "read4", '{"chapter_offset": 0}')
        tool_a.respond("First tool done")
        # tool_b left incomplete
        conversations.append(("not_ready_mixed", conv5))

        # Check which conversation is ready (should be earliest ready one)
        ready = Conversation.find_sendable_conversation(cursor)

        # Should get conv1 (earliest ready conversation)
        assert ready is not None
        assert ready.id == conv1.id

        # Complete the incomplete tools and verify state changes
        # Complete conv2's tool
        conv2_blocks = conv2.blocks
        incomplete_tool = next(
            b for b in conv2_blocks if b.tool_name and b.tool_response is None
        )
        incomplete_tool.respond("Now complete")

        # Complete conv5's remaining tool
        conv5_blocks = conv5.blocks
        incomplete_tool_b = next(b for b in conv5_blocks if b.tool_use_id == "read4")
        incomplete_tool_b.respond("Second tool done")

        # Check ready conversation again
        ready = Conversation.find_sendable_conversation(cursor)

        # Should still get conv1 (earliest ready conversation)
        assert ready is not None
        assert ready.id == conv1.id


def test_conversation_state_after_tool_errors(temp_db: SafeConnection) -> None:
    """Test conversation readiness after tool errors."""
    with temp_db.transaction_cursor() as cursor:
        conversation = Conversation.create(cursor)

        # Add user request
        conversation.add_user_text("Please read non-existent chapter")

        # Add tool use
        tool_block = conversation.add_tool_use(
            "ReadChapter", "error_tool", '{"chapter_offset": 999}'
        )

        # Initially not ready (incomplete tool)
        ready = Conversation.find_sendable_conversation(cursor)
        assert ready is None

        # Respond with error
        tool_block.respond_error("Chapter 999 does not exist")

        # Should now be ready (tool has response, even if error)
        ready = Conversation.find_sendable_conversation(cursor)
        assert ready is not None
        assert ready.id == conversation.id

        # Verify the error state is preserved
        blocks = conversation.blocks
        error_block = next(b for b in blocks if b.tool_name == "ReadChapter")
        assert error_block.errored is True
        assert error_block.tool_response == "Chapter 999 does not exist"


def test_conversation_lifecycle_state_transitions(temp_db: SafeConnection) -> None:
    """Test complete conversation lifecycle state transitions."""
    with temp_db.transaction_cursor() as cursor:
        Chapter.add_chapter(cursor, 1, ["Book"], "Content")

        conversation = Conversation.create(cursor)

        # State 1: Empty conversation (not ready)
        ready = Conversation.find_sendable_conversation(cursor)
        assert ready is None

        # State 2: User message added (ready)
        user_msg = conversation.add_user_text("Help me")
        ready = Conversation.find_sendable_conversation(cursor)
        assert ready is not None

        # State 3: Assistant responds with tool use (not ready - incomplete tool)
        conversation.add_assistant_text("I'll help you")
        tool_block = conversation.add_tool_use(
            "ReadChapter", "help1", '{"chapter_offset": 0}'
        )
        ready = Conversation.find_sendable_conversation(cursor)
        assert ready is None

        # State 4: Tool completes (ready again)
        tool_block.respond("Chapter content here")
        ready = Conversation.find_sendable_conversation(cursor)
        assert ready is not None
        assert ready.id == conversation.id

        # State 5: User message marked as sent
        # Note: Assistant messages are already sent=True by default,
        # and tool responses make the conversation ready
        user_msg.mark_as_sent()
        ready = Conversation.find_sendable_conversation(cursor)
        # Conversation is still ready because tools are complete
        assert ready is not None
        assert ready.id == conversation.id

        # State 6: New user message (ready again)
        conversation.add_user_text("Thanks, can you do more?")
        ready = Conversation.find_sendable_conversation(cursor)
        assert ready is not None
        assert ready.id == conversation.id

        # State 7: Multiple tool uses (not ready)
        conversation.add_assistant_text("Sure, let me use multiple tools")
        tool1 = conversation.add_tool_use(
            "ReadChapter", "more1", '{"chapter_offset": 0}'
        )
        tool2 = conversation.add_tool_use(
            "ReadChapter", "more2", '{"chapter_offset": 0}'
        )
        ready = Conversation.find_sendable_conversation(cursor)
        assert ready is None

        # State 8: Partial completion (still not ready)
        tool1.respond("First tool done")
        ready = Conversation.find_sendable_conversation(cursor)
        assert ready is None

        # State 9: Full completion (ready)
        tool2.respond("Second tool done")
        ready = Conversation.find_sendable_conversation(cursor)
        assert ready is not None


def test_conversation_parent_child_state_independence(temp_db: SafeConnection) -> None:
    """Test that parent and child conversation states are independent."""
    with temp_db.transaction_cursor() as cursor:
        # Create parent conversation
        parent_conv = Conversation.create(cursor)
        parent_conv.add_user_text("I need multiple agents")

        # Parent should be ready
        ready = Conversation.find_sendable_conversation(cursor)
        assert ready is not None
        assert ready.id == parent_conv.id

        # Spawn child from parent using a tool use block (SpawnAgent)
        spawn_block = parent_conv.add_tool_use(
            "SpawnAgent", "spawn1", '{"task": "Handle subtask"}'
        )
        child_conv = spawn_block.start_conversation()

        # Add tool response so parent remains ready
        spawn_block.respond('{"status": "spawned"}')

        # Parent is still ready (has unsent user message and completed tool)
        ready = Conversation.find_sendable_conversation(cursor)
        # Should get parent (earliest ready conversation)
        assert ready is not None
        assert ready.id == parent_conv.id

        # Add activity to child
        child_conv.add_user_text("Child agent task")

        # Both should now be ready, but we'll get parent (earliest)
        ready = Conversation.find_sendable_conversation(cursor)
        assert ready is not None
        assert ready.id == parent_conv.id  # Parent was created first

        # Add incomplete tool to child
        child_tool = child_conv.add_tool_use(
            "ReadChapter", "child1", '{"chapter_offset": 0}'
        )

        # Child should no longer be ready, parent still ready
        ready = Conversation.find_sendable_conversation(cursor)
        assert ready is not None
        assert ready.id == parent_conv.id

        # Mark parent user message and tool use as sent
        parent_blocks = parent_conv.blocks
        user_block = next(b for b in parent_blocks if b.text_role == "user")
        user_block.mark_as_sent()
        spawn_block.mark_as_sent()

        # Neither should be ready now
        ready = Conversation.find_sendable_conversation(cursor)
        assert ready is None

        # Complete child tool
        child_tool.respond("Child task complete")

        # Child should be ready again, parent still not
        ready = Conversation.find_sendable_conversation(cursor)
        assert ready is not None
        assert ready.id == child_conv.id


def test_conversation_cleanup_patterns(temp_db: SafeConnection) -> None:
    """Test patterns for conversation cleanup and state management."""
    with temp_db.transaction_cursor() as cursor:
        Chapter.add_chapter(cursor, 1, ["Book"], "Content")

        # Create conversations with various states
        conversations = []

        # Complete conversation (all sent)
        complete_conv = Conversation.create(cursor)
        user_msg = complete_conv.add_user_text("Complete request")
        complete_conv.add_assistant_text("Complete response")
        user_msg.mark_as_sent()
        conversations.append(("complete", complete_conv))

        # Active conversation (has unsent)
        active_conv = Conversation.create(cursor)
        active_conv.add_user_text("Active request")
        conversations.append(("active", active_conv))

        # Stuck conversation (incomplete tool)
        stuck_conv = Conversation.create(cursor)
        stuck_conv.add_user_text("Stuck request")
        stuck_conv.add_tool_use("ReadChapter", "stuck1", '{"chapter_offset": 0}')
        conversations.append(("stuck", stuck_conv))

        # Test identifying different conversation states
        # Ready conversations (should just be active)
        ready = Conversation.find_sendable_conversation(cursor)
        assert ready is not None
        assert ready.id == active_conv.id

        # All conversations - verify by counting the created conversations
        assert len(conversations) == 3

        # Conversations with unsent blocks - check using model properties
        active_has_unsent = any(not block.sent for block in active_conv.blocks)
        stuck_has_unsent = any(not block.sent for block in stuck_conv.blocks)
        complete_has_unsent = any(not block.sent for block in complete_conv.blocks)
        assert active_has_unsent
        assert stuck_has_unsent
        assert not complete_has_unsent

        # Conversations with incomplete tools - check using model properties
        stuck_has_incomplete = any(
            block.tool_name and block.tool_response is None
            for block in stuck_conv.blocks
        )
        active_has_incomplete = any(
            block.tool_name and block.tool_response is None
            for block in active_conv.blocks
        )
        complete_has_incomplete = any(
            block.tool_name and block.tool_response is None
            for block in complete_conv.blocks
        )
        assert stuck_has_incomplete
        assert not active_has_incomplete
        assert not complete_has_incomplete

        # Test conversation "completion" by marking all blocks as sent
        for _conv_type, conv in conversations:
            blocks = conv.blocks
            for block in blocks:
                if not block.sent and block.text_role:  # Only mark text blocks
                    block.mark_as_sent()

        # After marking all text blocks as sent, no conversations should be ready
        ready = Conversation.find_sendable_conversation(cursor)
        assert ready is None
