"""Tests for conversation routes in the web interface."""

import json

import pytest
from flask.testing import FlaskClient

from bookwiki.db import SafeConnection
from bookwiki.models import Block, Chapter, Conversation


@pytest.fixture
def test_app_conv(web_client: FlaskClient) -> FlaskClient:
    """Create Flask test app with conversations support."""
    return web_client


@pytest.fixture
def populated_conv_db(temp_db: SafeConnection) -> SafeConnection:
    """Populate database with conversations and blocks."""
    with temp_db.transaction_cursor() as cursor:
        # Add a chapter
        chapter = Chapter.add_chapter(
            cursor, 1, ["Book One", "Chapter 1"], "Chapter text..."
        )

        # Create root conversation
        conv1 = Conversation.create(cursor)

        # Mark the chapter as started by this conversation
        chapter.set_conversation_id(conv1)

        # Add blocks to conversation
        Block.create_text(
            cursor,
            conv1.id,
            conv1.current_generation,
            "user",
            "What happens in this chapter?",
        )
        Block.create_text(
            cursor,
            conv1.id,
            conv1.current_generation,
            "assistant",
            "Let me read the chapter...",
        )

        # Add a tool use block
        tool_block = Block.create_tool_use(
            cursor,
            conv1.id,
            conv1.current_generation,
            "ReadChapter",
            "tool_123",
            json.dumps({"chapter_offset": 0}),
        )

        # Add tool response
        tool_block.respond("Chapter content here...")

        # Add more conversation
        Block.create_text(
            cursor,
            conv1.id,
            conv1.current_generation,
            "assistant",
            "The chapter describes the beginning of the story.",
        )

        # Create a child conversation (spawned agent)
        spawn_block = Block.create_tool_use(
            cursor,
            conv1.id,
            conv1.current_generation,
            "SpawnAgent",
            "spawn_456",
            json.dumps({"task": "Analyze characters"}),
        )

        conv2 = Conversation.create(cursor, parent_block_id=spawn_block.id)
        Block.create_text(
            cursor,
            conv2.id,
            conv2.current_generation,
            "assistant",
            "Analyzing characters...",
        )

        # Create another root conversation
        conv3 = Conversation.create(cursor)
        Block.create_text(
            cursor, conv3.id, conv3.current_generation, "user", "Another conversation"
        )

        # Set one conversation as waiting
        conv3.set_waiting_on_id("waiting_123")

        # Update token counts
        conv1.update_tokens(100, 200)

    return temp_db


def test_conversations_list_empty(test_app_conv: FlaskClient) -> None:
    """Test conversations list with no conversations."""
    response = test_app_conv.get("/conversations")
    assert response.status_code == 200
    assert b"No conversations found" in response.data


def test_conversations_list_with_data(
    test_app_conv: FlaskClient,
    populated_conv_db: SafeConnection,  # noqa: ARG001
) -> None:
    """Test conversations list with multiple conversations."""
    response = test_app_conv.get("/conversations")
    assert response.status_code == 200

    # Should show root conversations only (not child) - now in table format
    assert b"#1" in response.data
    assert b"#3" in response.data
    # Look for conversation #2 specifically in a table cell context (not CSS)
    assert b'<td class="id-cell">#2</td>' not in response.data  # Child conversation

    # Check status indicators (badges in table)
    # Conv 1 has a SpawnAgent tool use without response
    assert b"Waiting Tools" in response.data  # Conv 1
    # Conv 3 has waiting_on_id set
    assert b"Waiting LLM" in response.data  # Conv 3

    # Check token display in table columns
    assert b"100" in response.data  # Input tokens
    assert b"200" in response.data  # Output tokens

    # Table should have proper headers
    assert b"<th" in response.data  # Table headers present


def test_conversation_detail(
    test_app_conv: FlaskClient,
    populated_conv_db: SafeConnection,  # noqa: ARG001
) -> None:
    """Test detailed conversation view."""
    response = test_app_conv.get("/conversation/1")
    assert response.status_code == 200

    # Check metadata
    assert b"Conversation #1" in response.data
    assert b"100 in" in response.data
    assert b"200 out" in response.data
    assert b"current" in response.data

    # Check blocks are displayed
    assert b"What happens in this chapter?" in response.data
    assert b"Let me read the chapter..." in response.data
    assert b"The chapter describes the beginning of the story." in response.data

    # Check tool use display
    assert b"ReadChapter" in response.data
    assert b'"chapter_offset": 0' in response.data

    # Check child conversations section
    assert b"#2" in response.data


def test_conversation_blocks_ordering(
    test_app_conv: FlaskClient,
    populated_conv_db: SafeConnection,  # noqa: ARG001
) -> None:
    """Test that blocks are displayed in chronological order."""
    response = test_app_conv.get("/conversation/1")
    assert response.status_code == 200

    content = response.data.decode("utf-8")

    # Find positions of text in the response
    pos1 = content.find("What happens in this chapter?")
    pos2 = content.find("Let me read the chapter...")
    pos3 = content.find("ReadChapter")
    pos4 = content.find("The chapter describes the beginning")

    # Verify chronological order
    assert pos1 < pos2 < pos3 < pos4


def test_conversation_tool_links(
    test_app_conv: FlaskClient,
    populated_conv_db: SafeConnection,  # noqa: ARG001
) -> None:
    """Test that tool uses generate appropriate links."""
    response = test_app_conv.get("/conversation/1")
    assert response.status_code == 200

    # Check that ReadChapter generates a link to the chapter
    assert b'href="/chapter/1"' in response.data
    assert b"View Chapter" in response.data


def test_conversation_child_display(
    test_app_conv: FlaskClient,
    populated_conv_db: SafeConnection,  # noqa: ARG001
) -> None:
    """Test that child conversations are properly displayed."""
    response = test_app_conv.get("/conversation/1")
    assert response.status_code == 200

    # Check table structure and child conversation reference
    assert b"<table" in response.data  # Check for basic table element
    assert b"#2" in response.data  # Child conversation ID should be displayed
    assert b"onclick=\"row_link(event, '/conversation/2')\"" in response.data


def test_conversation_deep_linking(
    test_app_conv: FlaskClient,
    populated_conv_db: SafeConnection,  # noqa: ARG001
) -> None:
    """Test deep linking to specific blocks."""
    # Get conversation detail page
    response = test_app_conv.get("/conversation/1")
    assert response.status_code == 200

    content = response.data.decode("utf-8")

    # Check that blocks have id attributes for deep linking
    assert 'id="block-' in content

    # Check that blocks have self-links
    assert 'href="#block-' in content


def test_conversation_not_found(test_app_conv: FlaskClient) -> None:
    """Test accessing non-existent conversation."""
    response = test_app_conv.get("/conversation/999")
    assert response.status_code == 404


def test_conversation_visual_indicators(
    test_app_conv: FlaskClient,
    populated_conv_db: SafeConnection,  # noqa: ARG001
) -> None:
    """Test visual indicators for conversation states."""
    # Test list view
    response = test_app_conv.get("/conversations")
    content = response.data.decode("utf-8")

    # Check for status indicators
    # Conv 3 has waiting_on_id
    assert "Waiting LLM" in content
    # Conv 1 has SpawnAgent tool without response
    assert "Waiting Tools" in content

    # Test detail view
    response = test_app_conv.get("/conversation/1")
    content = response.data.decode("utf-8")

    # Check for block type indicators (now using semantic elements)
    assert "User" in content or "Assistant" in content
    assert "<article" in content  # Check for article elements used for blocks
