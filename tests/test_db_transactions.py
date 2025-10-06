"""Database transaction tests for bookwiki SafeConnection."""

import sqlite3
import threading
import time

import pytest

from bookwiki.db import SafeConnection
from bookwiki.models import Conversation


def test_transaction_cursor_basic_usage(temp_db: SafeConnection) -> None:
    """Test basic transaction cursor functionality."""
    # Create a conversation within a transaction
    with temp_db.transaction_cursor() as cursor:
        conversation = Conversation.create(cursor)
        assert conversation.id is not None

        # Verify it exists within the transaction
        cursor.execute("SELECT COUNT(*) FROM conversation")
        count = cursor.fetchone()[0]
        assert count == 1

    # Verify it persists after transaction commits
    with temp_db.transaction_cursor() as cursor:
        cursor.execute("SELECT COUNT(*) FROM conversation")
        count = cursor.fetchone()[0]
        assert count == 1


def test_transaction_rollback_on_exception(temp_db: SafeConnection) -> None:
    """Test that transactions are rolled back on exceptions."""
    # First, create a baseline conversation
    with temp_db.transaction_cursor() as cursor:
        Conversation.create(cursor)

    # Verify baseline exists
    with temp_db.transaction_cursor() as cursor:
        cursor.execute("SELECT COUNT(*) FROM conversation")
        assert cursor.fetchone()[0] == 1

    # Now try to create another but raise an exception
    with pytest.raises(ValueError), temp_db.transaction_cursor() as cursor:
        Conversation.create(cursor)
        # Verify it was created within the transaction
        cursor.execute("SELECT COUNT(*) FROM conversation")
        assert cursor.fetchone()[0] == 2

        # Raise an exception to trigger rollback
        raise ValueError("Test exception")

    # Verify the second conversation was rolled back
    with temp_db.transaction_cursor() as cursor:
        cursor.execute("SELECT COUNT(*) FROM conversation")
        assert cursor.fetchone()[0] == 1  # Only the baseline remains


def test_concurrent_transaction_isolation(temp_db: SafeConnection) -> None:
    """Test that concurrent transactions don't interfere with each other."""
    results = []
    errors = []

    def create_conversations(thread_id: int, count: int) -> None:
        """Create conversations in a separate thread."""
        try:
            for i in range(count):
                with temp_db.transaction_cursor() as cursor:
                    conversation = Conversation.create(cursor)
                    # Add some token updates to make transactions longer
                    conversation.update_tokens(thread_id * 10 + i, thread_id * 5 + i)
                    results.append((thread_id, conversation.id))

                # Small delay to encourage interleaving
                time.sleep(0.001)
        except Exception as e:
            errors.append((thread_id, str(e)))

    # Start multiple threads creating conversations
    threads = []
    for thread_id in range(3):
        thread = threading.Thread(target=create_conversations, args=(thread_id, 5))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Check that no errors occurred
    assert len(errors) == 0, f"Errors occurred: {errors}"

    # Check that all conversations were created
    assert len(results) == 15  # 3 threads * 5 conversations each

    # Verify all conversations exist in database
    with temp_db.transaction_cursor() as cursor:
        cursor.execute("SELECT COUNT(*) FROM conversation")
        count = cursor.fetchone()[0]
        assert count == 15

    # Verify all conversation IDs are unique
    conversation_ids = [result[1] for result in results]
    assert len(conversation_ids) == len(set(conversation_ids))


def test_cursor_cleanup_on_exception(temp_db: SafeConnection) -> None:
    """Test that cursors are properly cleaned up even when exceptions occur."""
    # This test ensures that the context manager properly closes cursors
    # even when exceptions are raised

    # Note: cursor cleanup testing is implementation-dependent

    # Try to use transaction cursor with exception
    with (
        pytest.raises(RuntimeError, match="Test error"),
        temp_db.transaction_cursor() as cursor,
    ):
        Conversation.create(cursor)
        # This will succeed, but then we'll raise an exception
        raise RuntimeError("Test error")

    # Transaction cursor should still work after the exception
    with temp_db.transaction_cursor() as cursor:
        # This should succeed, proving cursor cleanup worked
        conversation = Conversation.create(cursor)
        assert conversation.id is not None

    # Verify the conversation was created
    with temp_db.transaction_cursor() as cursor:
        cursor.execute("SELECT COUNT(*) FROM conversation")
        count = cursor.fetchone()[0]
        # Only one conversation exists since the first transaction was rolled back
        # due to exception
        assert count == 1


def test_sequential_transaction_behavior(temp_db: SafeConnection) -> None:
    """Test sequential transaction behavior.

    Note: True nested transactions can cause deadlocks with SQLite's locking.
    This tests sequential transactions instead.
    """
    # Create first transaction
    with temp_db.transaction_cursor() as cursor1:
        Conversation.create(cursor1)
        cursor1.execute("SELECT COUNT(*) FROM conversation")
        count1 = cursor1.fetchone()[0]
        assert count1 == 1

    # Create second transaction (sequential, not nested)
    with temp_db.transaction_cursor() as cursor2:
        Conversation.create(cursor2)
        cursor2.execute("SELECT COUNT(*) FROM conversation")
        count2 = cursor2.fetchone()[0]
        assert count2 == 2

    # Verify final state
    with temp_db.transaction_cursor() as cursor:
        cursor.execute("SELECT COUNT(*) FROM conversation")
        count = cursor.fetchone()[0]
        assert count == 2


def test_transaction_with_complex_operations(temp_db: SafeConnection) -> None:
    """Test transactions with complex multi-table operations."""
    with temp_db.transaction_cursor() as cursor:
        # Create conversation
        conversation = Conversation.create(cursor)

        # Add multiple blocks
        conversation.add_user_text("Test message")
        conversation.add_assistant_text("Response")
        tool_block = conversation.add_tool_use(
            "TestTool", "test_id", '{"param": "value"}'
        )
        # Complete the tool use so conversation becomes ready
        tool_block.respond("Tool completed successfully")

        # Update tokens
        conversation.update_tokens(100, 50)

        # Verify everything exists within transaction
        cursor.execute("SELECT COUNT(*) FROM conversation")
        assert cursor.fetchone()[0] == 1

        cursor.execute("SELECT COUNT(*) FROM block")
        assert cursor.fetchone()[0] == 3

        cursor.execute(
            "SELECT total_input_tokens FROM conversation WHERE id = ?",
            (conversation.id,),
        )
        assert cursor.fetchone()[0] == 100

    # Verify everything persisted after transaction
    with temp_db.transaction_cursor() as cursor:
        cursor.execute("SELECT COUNT(*) FROM conversation")
        assert cursor.fetchone()[0] == 1

        cursor.execute("SELECT COUNT(*) FROM block")
        assert cursor.fetchone()[0] == 3

        # Verify conversation can still be retrieved and used
        ready_conversation = Conversation.find_sendable_conversation(cursor)
        assert ready_conversation is not None

        retrieved_conversation = ready_conversation
        blocks = retrieved_conversation.blocks
        assert len(blocks) == 3


def test_database_consistency_patterns(temp_db: SafeConnection) -> None:
    """Test basic database consistency patterns.

    Note: Complex concurrent testing with SQLite can cause deadlocks
    due to transaction isolation. This test focuses on basic consistency.
    """
    with temp_db.transaction_cursor() as cursor:
        # Create multiple conversations sequentially to test consistency
        conversations = []
        for i in range(3):
            conversation = Conversation.create(cursor)
            conversation.add_user_text(f"Message from iteration {i}")
            conversation.update_tokens(i * 10, i * 5)
            conversations.append(conversation)

        # Verify all conversations exist
        cursor.execute("SELECT COUNT(*) FROM conversation")
        conversation_count = cursor.fetchone()[0]
        assert conversation_count == 3

        cursor.execute("SELECT COUNT(*) FROM block")
        block_count = cursor.fetchone()[0]
        assert block_count == 3  # One user message per conversation

        # Verify token totals are correct
        cursor.execute(
            "SELECT SUM(total_input_tokens), SUM(total_output_tokens) FROM conversation"
        )
        total_input, total_output = cursor.fetchone()
        expected_input = sum(i * 10 for i in range(3))  # 0+10+20 = 30
        expected_output = sum(i * 5 for i in range(3))  # 0+5+10 = 15
        assert total_input == expected_input
        assert total_output == expected_output


def test_cursor_usage_after_context_exit(temp_db: SafeConnection) -> None:
    """Test that using a cursor after exiting transaction_cursor raises an exception."""
    saved_cursor = None

    # Create a cursor within the context and save it
    with temp_db.transaction_cursor() as cursor:
        saved_cursor = cursor
        # Verify cursor works within context
        cursor.execute("SELECT 1")
        assert cursor.fetchone()[0] == 1

    # Now try to use the cursor after context exit
    with pytest.raises(sqlite3.ProgrammingError):
        saved_cursor.execute("SELECT 1")

    # Also test that fetchone raises after context exit
    with temp_db.transaction_cursor() as cursor:
        cursor.execute("SELECT 1")
        saved_cursor = cursor

    with pytest.raises(sqlite3.ProgrammingError):
        saved_cursor.fetchone()
