"""Domain-aware test helper functions for FanWiki tests.

This module provides high-level abstractions for test data creation that follow
the same patterns as production code, eliminating the need for raw SQL in tests.
"""

import json
from sqlite3 import Cursor
from typing import Optional

from bookwiki.models import Block, Conversation


def verify_conversation_token_state(
    cursor: Cursor,
    conversation: Conversation,
    expected_input_tokens: int,
    expected_output_tokens: int,
    expected_current_tokens: int,
) -> None:
    """Verify a conversation's token counts by reloading from database.

    This replaces raw SQL verification queries with a high-level approach.

    Args:
        cursor: Database cursor
        conversation: Conversation to verify
        expected_input_tokens: Expected total input tokens
        expected_output_tokens: Expected total output tokens
        expected_current_tokens: Expected current tokens
    """
    # Reload conversation from database to get fresh state
    reloaded = Conversation.get_by_id(cursor, conversation.id)
    assert reloaded is not None, f"Conversation {conversation.id} not found"

    assert reloaded.total_input_tokens == expected_input_tokens
    assert reloaded.total_output_tokens == expected_output_tokens
    assert reloaded.current_tokens == expected_current_tokens


def verify_all_blocks_sent(cursor: Cursor, conversation: Conversation) -> None:
    """Verify that all blocks in a conversation are marked as sent.

    This replaces raw SQL verification with high-level model access.

    Args:
        cursor: Database cursor
        conversation: Conversation to verify
    """
    # Reload conversation to clear cached properties
    reloaded = Conversation.get_by_id(cursor, conversation.id)
    assert reloaded is not None

    # Use the model's blocks property instead of raw SQL
    all_blocks = reloaded.blocks
    for block in all_blocks:
        assert block.sent is True, f"Block {block.id} is not marked as sent"


def verify_conversation_waiting_state(
    cursor: Cursor,
    conversation: Conversation,
    expected_waiting_on_id: Optional[str],
) -> None:
    """Verify a conversation's waiting_on_id state by reloading from database.

    This replaces raw SQL verification queries with high-level model access.

    Args:
        cursor: Database cursor
        conversation: Conversation to verify
        expected_waiting_on_id: Expected waiting_on_id value (None for not waiting)
    """
    # Reload conversation from database to get fresh state
    reloaded = Conversation.get_by_id(cursor, conversation.id)
    assert reloaded is not None, f"Conversation {conversation.id} not found"

    assert reloaded.waiting_on_id == expected_waiting_on_id


def create_feedback_request_block(
    conversation: Conversation,
    request_text: str,
    use_id: str = "feedback_test",
    already_sent: bool = True,
    response_text: Optional[str] = None,
) -> Block:
    """Create a RequestExpertFeedback tool use block with optional response.

    This replaces raw SQL block creation with high-level model methods.

    Args:
        conversation: The conversation to add the feedback request to
        request_text: The feedback request text
        use_id: Tool use ID (defaults to "feedback_test")
        already_sent: Whether the block should be marked as sent
        response_text: Optional response text (for pre-responded feedback)

    Returns:
        The created feedback request block
    """

    params = json.dumps({"request": request_text})
    block = conversation.add_tool_use("RequestExpertFeedback", use_id, params)

    if response_text:
        block.respond(response_text)

    if already_sent:
        block.mark_as_sent()

    return block


def verify_block_sent_state(cursor: Cursor, block: Block, expected_sent: bool) -> None:
    """Verify a block's sent state by reloading from database.

    This replaces raw SQL verification queries with high-level model access.

    Args:
        cursor: Database cursor
        block: Block to verify
        expected_sent: Expected sent state
    """
    # Reload block from database to get fresh state
    reloaded = Block.get_by_id(cursor, block.id)
    assert reloaded is not None, f"Block {block.id} not found"

    assert reloaded.sent is expected_sent


def verify_block_tool_response(
    cursor: Cursor, block: Block, expected_response: Optional[str]
) -> None:
    """Verify a block's tool_response by reloading from database.

    This replaces raw SQL verification queries with high-level model access.

    Args:
        cursor: Database cursor
        block: Block to verify
        expected_response: Expected tool response (None if no response)
    """
    # Reload block from database to get fresh state
    reloaded = Block.get_by_id(cursor, block.id)
    assert reloaded is not None, f"Block {block.id} not found"

    assert reloaded.tool_response == expected_response


def verify_block_tool_response_by_id(
    cursor: Cursor, block_id: int, expected_response: Optional[str]
) -> None:
    """Verify a block's tool_response by ID, reloading from database.

    This replaces raw SQL verification queries with high-level model access.

    Args:
        cursor: Database cursor
        block_id: ID of the block to verify
        expected_response: Expected tool response (None if no response)
    """
    # Load block from database
    reloaded = Block.get_by_id(cursor, block_id)
    assert reloaded is not None, f"Block {block_id} not found"

    assert reloaded.tool_response == expected_response
