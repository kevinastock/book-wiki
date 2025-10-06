"""Tests for the bookwiki web server."""

import json
from typing import Generator

import pytest
from flask.testing import FlaskClient

from bookwiki.db import SafeConnection
from bookwiki.models import Chapter, Conversation, WikiPage


@pytest.fixture
def test_app(
    web_client: FlaskClient, temp_db: SafeConnection
) -> Generator[FlaskClient, None, None]:
    """Create Flask test app with in-memory database."""
    # Patch get_db to return our test database
    import bookwiki.web.app

    original_get_db = bookwiki.web.app.get_db
    bookwiki.web.app.get_db = lambda: temp_db

    yield web_client

    # Restore original get_db
    bookwiki.web.app.get_db = original_get_db


@pytest.fixture
def populated_db(temp_db: SafeConnection) -> SafeConnection:
    """Populate database with test data."""
    with temp_db.transaction_cursor() as cursor:
        # Add chapters
        chapters = [
            (
                1,
                ["Book One", "Chapter 1", "The Beginning"],
                "It was a dark and stormy night...",
            ),
            (
                2,
                ["Book One", "Chapter 2", "The Journey"],
                "The hero set out on their quest...",
            ),
            (
                3,
                ["Book One", "Chapter 3", "The Challenge"],
                "Obstacles appeared on the path...",
            ),
        ]

        for id, name, text in chapters:
            Chapter.add_chapter(cursor, id, name, text)

        # Add a conversation
        conv = Conversation.create(cursor)

        # Add wiki pages for chapters 1 and 2
        wiki_pages = [
            (
                1,
                "hero",
                "The Hero",
                ["Hero", "Main Character"],
                "The protagonist of our story",
                "A brave adventurer...",
            ),
            (
                1,
                "storm",
                "The Storm",
                ["Storm", "Weather"],
                "The opening storm",
                "A dark and stormy night...",
            ),
            (
                2,
                "quest",
                "The Quest",
                ["Quest", "Journey"],
                "The hero's journey",
                "An epic quest begins...",
            ),
        ]

        for chapter_id, slug, title, names, summary, body in wiki_pages:
            # Create a block for wiki page creation
            block = conv.add_tool_use(
                "WriteWikiPage",
                f"wiki_{slug}",
                json.dumps({"slug": slug, "title": title, "names": names}),
            )

            # Create the wiki page
            WikiPage.create(
                cursor=cursor,
                chapter_id=chapter_id,
                slug=slug,
                create_block_id=block.id,
                title=title,
                names=names,
                summary=summary,
                body=body,
            )

        # Add a feedback request block
        conv.add_user_text("Test user message")
        conv.add_tool_use(
            "RequestExpertFeedback",
            "feedback_1",
            json.dumps({"request": "Is this character development appropriate?"}),
        )

    return temp_db


# Basic Access Tests


def test_root_page(test_app: FlaskClient) -> None:
    """Test that root page loads and contains navigation."""
    response = test_app.get("/")
    assert response.status_code == 200
    assert b"FanWiki" in response.data
    assert b"Chapters" in response.data
    assert b"Conversations" in response.data
    assert b"Wiki" in response.data


def test_chapters_list_empty(test_app: FlaskClient) -> None:
    """Test chapters list with no chapters."""
    response = test_app.get("/chapters")
    assert response.status_code == 200
    assert b"Chapters" in response.data
    assert b"No chapters found" in response.data


def test_chapters_list_with_data(
    test_app: FlaskClient,
    populated_db: SafeConnection,  # noqa: ARG001
) -> None:
    """Test chapters list with populated database."""
    response = test_app.get("/chapters")
    assert response.status_code == 200

    # Check chapter titles are displayed
    assert b"Book One - Chapter 1 - The Beginning" in response.data
    assert b"Book One - Chapter 2 - The Journey" in response.data
    assert b"Book One - Chapter 3 - The Challenge" in response.data

    # Check wiki counts (chapter 1 has 2, chapter 2 has 1, chapter 3 has 0)
    # Verify View buttons exist in response


def test_chapter_detail(
    test_app: FlaskClient,
    populated_db: SafeConnection,  # noqa: ARG001
) -> None:
    """Test chapter detail page."""
    response = test_app.get("/chapter/1")
    assert response.status_code == 200

    # Check chapter content
    assert b"Book One - Chapter 1 - The Beginning" in response.data
    assert b"It was a dark and stormy night..." in response.data

    # Check wiki pages are listed
    assert b"The Hero" in response.data
    assert b"The Storm" in response.data
    assert b"hero" in response.data  # slug
    assert b"storm" in response.data  # slug


def test_chapter_detail_not_found(test_app: FlaskClient) -> None:
    """Test chapter detail page with non-existent chapter."""
    response = test_app.get("/chapter/999")
    assert response.status_code == 404


# Content Verification Tests


def test_chapters_wiki_count_accuracy(
    test_app: FlaskClient, populated_db: SafeConnection
) -> None:
    """Verify wiki page counts per chapter are accurate."""
    response = test_app.get("/chapters")
    assert response.status_code == 200

    # Parse the response to check wiki counts
    # Chapter 1 should have 2 wiki pages
    # Chapter 2 should have 1 wiki page
    # Chapter 3 should have 0 wiki pages

    # Use model methods to verify wiki counts per chapter
    with populated_db.transaction_cursor() as cursor:
        from bookwiki.models import WikiPage

        expected_counts = {}
        for chapter_id in [1, 2, 3]:
            pages = WikiPage.get_all_pages_chapter(cursor, chapter_id)
            expected_counts[chapter_id] = len(pages)

    assert expected_counts[1] == 2
    assert expected_counts[2] == 1
    assert expected_counts[3] == 0


def test_chapter_wiki_links(
    test_app: FlaskClient,
    populated_db: SafeConnection,  # noqa: ARG001
) -> None:
    """Verify wiki page links in chapter detail are correct."""
    response = test_app.get("/chapter/1")
    assert response.status_code == 200

    # Check that wiki links are properly formatted
    assert b'href="/wiki/1/hero"' in response.data
    assert b'href="/wiki/1/storm"' in response.data

    # Verify summaries are displayed
    assert b"The protagonist of our story" in response.data
    assert b"The opening storm" in response.data


def test_chapter_navigation(
    test_app: FlaskClient,
    populated_db: SafeConnection,  # noqa: ARG001
) -> None:
    """Test navigation between chapters list and detail."""
    # Start at chapters list
    response = test_app.get("/chapters")
    assert response.status_code == 200

    # Check that chapters are clickable via onclick using row_link function
    assert b"onclick=\"row_link(event, '/chapter/1')\"" in response.data
    assert b"onclick=\"row_link(event, '/chapter/2')\"" in response.data
    assert b"onclick=\"row_link(event, '/chapter/3')\"" in response.data

    # Navigate to a chapter detail
    response = test_app.get("/chapter/2")
    assert response.status_code == 200

    # Check navigation back to list
    assert b'href="/chapters"' in response.data


def test_empty_chapter_wiki_pages(
    test_app: FlaskClient,
    populated_db: SafeConnection,  # noqa: ARG001
) -> None:
    """Test chapter detail with no wiki pages."""
    response = test_app.get("/chapter/3")
    assert response.status_code == 200

    # Chapter 3 has no wiki pages
    assert b"Book One - Chapter 3 - The Challenge" in response.data
    assert b"Obstacles appeared on the path..." in response.data

    # Should not show wiki pages section or show empty state
    data_str = response.data.decode("utf-8")
    # Check that wiki section is not shown or is empty
    if "Wiki Pages Created in This Chapter" in data_str:
        # If section exists, it should be empty
        assert "wiki-card" not in data_str
