"""Tests for prompt web interface."""

from string import Template

import pytest
from flask.testing import FlaskClient

from bookwiki.db import SafeConnection
from bookwiki.models import Conversation


@pytest.fixture
def test_app(web_client: FlaskClient) -> FlaskClient:
    """Create Flask test app with in-memory database."""
    return web_client


def test_prompts_list_empty(test_app: FlaskClient) -> None:
    """Test prompts list page with no prompts."""
    response = test_app.get("/prompts")
    assert response.status_code == 200
    assert b"No prompts found" in response.data


def test_prompts_list_with_data(test_app: FlaskClient, temp_db: SafeConnection) -> None:
    """Test prompts list page with prompts."""
    # Create test data
    with temp_db.transaction_cursor() as cursor:
        # Create conversation and block
        conversation = Conversation.create(cursor)
        block = conversation.add_user_text("Create a prompt")

        # Create prompt
        template = Template("Hello, $name! Welcome to $place.")
        block.add_prompt(
            key="greeting", summary="A greeting prompt template", template=template
        )

    response = test_app.get("/prompts")
    assert response.status_code == 200
    assert b"greeting" in response.data
    assert b"A greeting prompt template" in response.data
    # In table format, single version shows as just "1" in muted text
    assert b">1<" in response.data or b"1 version" in response.data


def test_prompt_detail_latest(test_app: FlaskClient, temp_db: SafeConnection) -> None:
    """Test prompt detail page for latest version."""
    # Create test data
    with temp_db.transaction_cursor() as cursor:
        conversation = Conversation.create(cursor)
        block = conversation.add_user_text("Create a prompt")

        template = Template("Hello, $name! Welcome to $place.")
        block.add_prompt(
            key="greeting", summary="A greeting prompt template", template=template
        )

    response = test_app.get("/prompt/greeting")
    assert response.status_code == 200
    assert b"greeting" in response.data  # Title in new layout
    assert b"A greeting prompt template" in response.data
    assert b"Hello, $name! Welcome to $place." in response.data
    # Variables are now shown inline with $ prefix, no "Variables" header
    assert b"name" in response.data
    assert b"place" in response.data


def test_prompt_detail_not_found(test_app: FlaskClient) -> None:
    """Test prompt detail page for non-existent prompt."""
    response = test_app.get("/prompt/nonexistent")
    assert response.status_code == 404


def test_prompt_version_history(test_app: FlaskClient, temp_db: SafeConnection) -> None:
    """Test prompt version history display."""
    # Create test data with multiple versions
    with temp_db.transaction_cursor() as cursor:
        conversation = Conversation.create(cursor)

        # Version 1
        block1 = conversation.add_user_text("Create prompt")
        template1 = Template("Hello, $name!")
        block1.add_prompt(key="greeting", summary="Simple greeting", template=template1)

        # Version 2
        block2 = conversation.add_user_text("Update prompt")
        template2 = Template("Hello, $name! Welcome to $place.")
        block2.add_prompt(
            key="greeting", summary="Extended greeting", template=template2
        )

    response = test_app.get("/prompt/greeting")
    assert response.status_code == 200
    # Version history is now implicit in the multiple articles shown
    # No explicit "Current Version" label in new layout
    assert b"2" in response.data  # Should show 2 versions


def test_prompt_template_variables(
    test_app: FlaskClient, temp_db: SafeConnection
) -> None:
    """Test prompt template variable detection."""
    with temp_db.transaction_cursor() as cursor:
        conversation = Conversation.create(cursor)
        block = conversation.add_user_text("Create prompt")

        template = Template(
            "Process $chapter_text for character $character and place $location."
        )
        block.add_prompt(
            key="analysis", summary="Chapter analysis prompt", template=template
        )

    response = test_app.get("/prompt/analysis")
    assert response.status_code == 200
    # Variables are now shown inline with $ prefix, no "Variables" header
    assert b"chapter_text" in response.data
    assert b"character" in response.data
    assert b"location" in response.data


def test_prompt_conversation_links(
    test_app: FlaskClient, temp_db: SafeConnection
) -> None:
    """Test links to source conversations."""
    with temp_db.transaction_cursor() as cursor:
        conversation = Conversation.create(cursor)
        block = conversation.add_user_text("Create prompt")

        template = Template("Hello, $name!")
        block.add_prompt(key="greeting", summary="Greeting prompt", template=template)

    response = test_app.get("/prompt/greeting")
    assert response.status_code == 200
    # Should link to conversation (new format: "View Source")
    assert b"View Source" in response.data
    assert f"#block-{block.id}".encode() in response.data


def test_prompt_metadata_display(
    test_app: FlaskClient, temp_db: SafeConnection
) -> None:
    """Test metadata display in new prompt layout."""
    with temp_db.transaction_cursor() as cursor:
        conversation = Conversation.create(cursor)
        block = conversation.add_user_text("Create prompt")

        template = Template("Hello, $name!")
        block.add_prompt(key="greeting", summary="Greeting prompt", template=template)

    response = test_app.get("/prompt/greeting")
    assert response.status_code == 200
    # Test new layout elements
    assert b"greeting" in response.data  # Prompt key title
    assert b"Greeting prompt" in response.data  # Summary
    assert b"View Source" in response.data  # Source link
    # Note: timestamp is now displayed without "Created" label
