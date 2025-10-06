"""Tests for feedback web interface."""

import pytest
from flask.testing import FlaskClient

from bookwiki.db import SafeConnection
from bookwiki.models import Conversation
from tests.helpers import (
    create_feedback_request_block,
    verify_block_tool_response_by_id,
)


@pytest.fixture
def test_app(web_client: FlaskClient) -> FlaskClient:
    """Create Flask test app with in-memory database."""
    return web_client


def test_feedback_empty(test_app: FlaskClient) -> None:
    """Test feedback page with no feedback requests."""
    response = test_app.get("/feedback")
    assert response.status_code == 200
    assert b"No feedback requests found" in response.data


def test_feedback_with_requests(test_app: FlaskClient, temp_db: SafeConnection) -> None:
    """Test feedback page with feedback requests."""
    with temp_db.transaction_cursor() as cursor:
        # Create a conversation
        conversation = Conversation.create(cursor)

        # Create feedback request blocks using helper
        create_feedback_request_block(
            conversation,
            "How should I handle character X's development? Is the pacing appropriate?",
            "feedback-1",
        )

        create_feedback_request_block(
            conversation,
            "Should I include more world-building details?",
            "feedback-2",
        )

    response = test_app.get("/feedback")
    assert response.status_code == 200
    assert b"Feedback Request #" in response.data
    assert b"How should I handle character X" in response.data
    assert b"Is the pacing appropriate?" in response.data
    assert b"Should I include more world-building" in response.data
    # Context is no longer displayed


def test_feedback_request_parsing(
    test_app: FlaskClient, temp_db: SafeConnection
) -> None:
    """Test parsing of different request formats."""
    with temp_db.transaction_cursor() as cursor:
        # Create a conversation
        conversation = Conversation.create(cursor)

        # Create feedback request with no request text using helper
        create_feedback_request_block(conversation, "", "feedback-1")

        # Create another feedback request with different content
        create_feedback_request_block(conversation, "Test request", "feedback-2")

    response = test_app.get("/feedback")
    assert response.status_code == 200
    # Both feedback requests should appear
    assert response.data.count(b"Feedback Request #") == 2


def test_feedback_conversation_links(
    test_app: FlaskClient, temp_db: SafeConnection
) -> None:
    """Test links to source conversations."""
    with temp_db.transaction_cursor() as cursor:
        # Create a conversation
        conversation = Conversation.create(cursor)

        # Create feedback request using helper
        feedback_block = create_feedback_request_block(
            conversation, "Test question", "feedback-1", already_sent=True
        )
        block_id = feedback_block.id

    response = test_app.get("/feedback")
    assert response.status_code == 200

    # Check that conversation link is present with deep link to block
    expected_link = f"/conversation/{conversation.id}#block-{block_id}"
    assert expected_link.encode() in response.data


def test_feedback_multiline_request_display(
    test_app: FlaskClient, temp_db: SafeConnection
) -> None:
    """Test display of multiline request text."""
    with temp_db.transaction_cursor() as cursor:
        # Create a conversation
        conversation = Conversation.create(cursor)

        # Create feedback request with multiline request using helper
        request_text = (
            "This is a multiline request.\n"
            "It has several lines.\n"
            "Each should be properly formatted."
        )
        create_feedback_request_block(
            conversation, request_text, "feedback-1", already_sent=True
        )

    response = test_app.get("/feedback")
    assert response.status_code == 200
    assert b"<p>This is a multiline request." in response.data
    assert b"It has several lines." in response.data


def test_feedback_submission(test_app: FlaskClient, temp_db: SafeConnection) -> None:
    """Test submitting feedback responses."""
    with temp_db.transaction_cursor() as cursor:
        # Create a conversation
        conversation = Conversation.create(cursor)

        # Create feedback request using helper
        feedback_block = create_feedback_request_block(
            conversation, "Test question", "feedback-1", already_sent=True
        )
        block_id = feedback_block.id
        conversation_id = conversation.id

    # Test successful submission
    response = test_app.post(
        f"/feedback/{block_id}/submit",
        data={"response": "This is my feedback response"},
        follow_redirects=False,
    )
    assert response.status_code == 302  # Redirect after submission
    assert response.location.endswith("/feedback")

    # Verify the response was stored using helper
    with temp_db.transaction_cursor() as cursor:
        verify_block_tool_response_by_id(
            cursor, block_id, "This is my feedback response"
        )

    # Test submitting to already responded request
    response = test_app.post(
        f"/feedback/{block_id}/submit",
        data={"response": "Another response"},
        follow_redirects=False,
    )
    assert response.status_code == 302  # Redirects even on error
    assert response.location.endswith("/feedback")

    # Test empty response
    with temp_db.transaction_cursor() as cursor:
        # Get the conversation again and create another feedback request
        reloaded_conversation = Conversation.get_by_id(cursor, conversation_id)
        assert reloaded_conversation is not None
        feedback_block2 = create_feedback_request_block(
            reloaded_conversation, "Another question", "feedback-2", already_sent=True
        )
        block_id2 = feedback_block2.id

    response = test_app.post(
        f"/feedback/{block_id2}/submit",
        data={"response": "   "},  # Only whitespace
        follow_redirects=False,
    )
    assert response.status_code == 302  # Redirects even on error
    assert response.location.endswith("/feedback")

    # Test missing response field
    response = test_app.post(
        f"/feedback/{block_id2}/submit",
        data={},
        follow_redirects=False,
    )
    assert response.status_code == 302  # Redirects even on error
    assert response.location.endswith("/feedback")

    # Test nonexistent block
    response = test_app.post(
        "/feedback/99999/submit",
        data={"response": "Test response"},
        follow_redirects=False,
    )
    assert response.status_code == 302  # Redirects even on error
    assert response.location.endswith("/feedback")


def test_feedback_filtering_responded_requests(
    test_app: FlaskClient, temp_db: SafeConnection
) -> None:
    """Test that responded feedback requests are not shown."""
    with temp_db.transaction_cursor() as cursor:
        conversation = Conversation.create(cursor)

        # Create two feedback requests - one responded, one not using helpers
        create_feedback_request_block(
            conversation,
            "Responded question",
            "feedback-responded",
            already_sent=True,
            response_text="This has been responded to",
        )

        create_feedback_request_block(
            conversation, "Pending question", "feedback-pending", already_sent=True
        )

    response = test_app.get("/feedback")
    assert response.status_code == 200

    # Should only show the pending request
    assert b"Pending question" in response.data
    assert b"Responded question" not in response.data


def test_feedback_ordering(test_app: FlaskClient, temp_db: SafeConnection) -> None:
    """Test that feedback requests are ordered by creation time (newest first)."""
    with temp_db.transaction_cursor() as cursor:
        # Create a conversation
        conversation = Conversation.create(cursor)

        # Create two feedback requests - order matters for this test
        create_feedback_request_block(
            conversation, "Older question", "feedback-1", already_sent=True
        )

        create_feedback_request_block(
            conversation, "Newer question", "feedback-2", already_sent=True
        )

    response = test_app.get("/feedback")
    assert response.status_code == 200

    # Find positions of the requests to verify ordering
    response_text = response.data.decode()
    newer_pos = response_text.find("Newer question")
    older_pos = response_text.find("Older question")

    # Newer request should appear first (lower position)
    assert newer_pos < older_pos
    assert newer_pos != -1 and older_pos != -1
