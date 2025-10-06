"""Tests for wiki routes in the web interface."""

import pytest
from flask.testing import FlaskClient

from bookwiki.db import SafeConnection
from bookwiki.models import Block, Chapter, Conversation, WikiPage


@pytest.fixture
def test_app_wiki(web_client: FlaskClient) -> FlaskClient:
    """Create Flask test app with wiki support."""
    return web_client


@pytest.fixture
def populated_wiki_db(temp_db: SafeConnection) -> SafeConnection:
    """Populate database with wiki pages for testing."""
    with temp_db.transaction_cursor() as cursor:
        # Add chapters (using 0-based indexing to match array positions)
        Chapter.add_chapter(cursor, 0, ["Book One", "Chapter 1"], "Chapter 1 text...")
        Chapter.add_chapter(cursor, 1, ["Book One", "Chapter 2"], "Chapter 2 text...")
        Chapter.add_chapter(cursor, 2, ["Book One", "Chapter 3"], "Chapter 3 text...")

        # Add conversations and blocks for wiki page creation
        conv1_id = Conversation.create(cursor)
        conv2_id = Conversation.create(cursor)

        # Mark chapters as started by conversations
        chapter0 = Chapter.read_chapter(cursor, 0)
        chapter1 = Chapter.read_chapter(cursor, 1)
        chapter2 = Chapter.read_chapter(cursor, 2)
        assert chapter0 and chapter1 and chapter2

        chapter0.set_conversation_id(conv1_id)
        chapter1.set_conversation_id(conv2_id)
        chapter2.set_conversation_id(conv1_id)

        # Create some blocks
        block1 = Block.create_text(
            cursor,
            conv1_id.id,
            conv1_id.current_generation,
            "user",
            "Create wiki page for Gandalf",
        )
        block2 = Block.create_text(
            cursor,
            conv2_id.id,
            conv2_id.current_generation,
            "user",
            "Create wiki page for Frodo",
        )

        # Mark blocks as sent
        block1.mark_as_sent()
        block2.mark_as_sent()

        # Create wiki pages following proper chapter progression
        # Chapter 0: Start with Gandalf and Bilbo
        WikiPage.create(
            cursor,
            chapter_id=0,
            slug="gandalf",
            create_block_id=block1.id,
            title="Gandalf the Grey",
            names=["Gandalf", "Gandalf the Grey"],
            summary="A wise wizard who guides the Fellowship.",
            body=(
                "Gandalf is a powerful wizard known for his wisdom and "
                "magical abilities.\n"
                "He carries a staff and has a long grey beard."
            ),
        )

        WikiPage.create(
            cursor,
            chapter_id=0,
            slug="bilbo-baggins",
            create_block_id=block1.id,
            title="Bilbo Baggins",
            names=["Bilbo", "Bilbo Baggins"],
            summary="Former Ring-bearer, now retired in Rivendell.",
            body=(
                "Bilbo is Frodo's cousin who previously carried the Ring for "
                "many years."
            ),
        )

        # Chapter 1: Copy previous pages and add Frodo, update Bilbo
        WikiPage.copy_current_for_new_chapter(cursor, 1)

        WikiPage.create(
            cursor,
            chapter_id=1,
            slug="frodo-baggins",
            create_block_id=block2.id,
            title="Frodo Baggins",
            names=["Frodo", "Frodo Baggins", "Mr. Frodo"],
            summary="A hobbit from the Shire who carries the Ring.",
            body=(
                "Frodo is a hobbit who inherited the One Ring from Bilbo.\n"
                "He embarks on a quest to destroy it in Mount Doom."
            ),
        )

        WikiPage.create(
            cursor,
            chapter_id=1,
            slug="bilbo-baggins",
            create_block_id=block2.id,
            title="Bilbo Baggins",
            names=["Bilbo", "Bilbo Baggins"],
            summary="Former Ring-bearer, now retired in Rivendell.",
            body=(
                "Bilbo is Frodo's cousin who previously carried the Ring for "
                "many years.\n"
                "He has now departed to the Undying Lands."
            ),
        )

        # Chapter 2: Copy previous pages and update Gandalf
        WikiPage.copy_current_for_new_chapter(cursor, 2)

        WikiPage.create(
            cursor,
            chapter_id=2,
            slug="gandalf",
            create_block_id=block1.id,
            title="Gandalf the White",
            names=["Gandalf", "Gandalf the White", "Mithrandir"],
            summary=(
                "A wise wizard who guides the Fellowship, "
                "now returned as Gandalf the White."
            ),
            body=(
                "Gandalf has returned as Gandalf the White after his battle with "
                "the Balrog.\n"
                "He is now even more powerful and continues to guide the Fellowship."
            ),
        )

    return temp_db


def test_wiki_list_empty(test_app_wiki: FlaskClient) -> None:
    """Test wiki list with no wiki pages."""
    response = test_app_wiki.get("/wiki")
    # With no started chapters, should return 404
    assert response.status_code == 404


def test_wiki_list_with_data(
    test_app_wiki: FlaskClient,
    populated_wiki_db: SafeConnection,  # noqa: ARG001
) -> None:
    """Test wiki list with wiki pages."""
    response = test_app_wiki.get("/wiki")
    # Should redirect to latest started chapter
    assert response.status_code == 302

    # Follow the redirect to test actual list functionality
    response = test_app_wiki.get("/wiki", follow_redirects=True)
    assert response.status_code == 200

    # Should show all wiki page names/slugs (no alphabetical grouping)
    assert b"bilbo-baggins" in response.data
    assert b"frodo-baggins" in response.data
    assert b"gandalf" in response.data

    # Should show summaries
    assert b"Former Ring-bearer, now retired in Rivendell." in response.data
    assert b"A hobbit from the Shire who carries the Ring." in response.data
    assert (
        b"A wise wizard who guides the Fellowship, now returned as Gandalf the White."
        in response.data
    )


def test_wiki_page_display(
    test_app_wiki: FlaskClient,
    populated_wiki_db: SafeConnection,  # noqa: ARG001
) -> None:
    """Test wiki page display."""
    # Test Gandalf at chapter 1 (should show v1 - Grey)
    response = test_app_wiki.get("/wiki/0/gandalf")
    assert response.status_code == 200
    assert b"Gandalf the Grey" in response.data
    assert b"A wise wizard who guides the Fellowship." in response.data
    assert b"long grey beard" in response.data

    # Test for names/aliases
    assert b"Known as:" in response.data
    assert b"code>Gandalf</code>" in response.data
    assert b"code>Gandalf the Grey</code>" in response.data

    # Should show chapter info
    assert b"Chapter 1" in response.data

    # Test Gandalf at chapter 3 (should show v2 - White)
    response = test_app_wiki.get("/wiki/2/gandalf")
    assert response.status_code == 200
    assert b"Gandalf the White" in response.data
    assert b"returned as Gandalf the White" in response.data
    assert b"battle with the Balrog" in response.data
    # Test for names in chapter 3 version
    assert b"code>Gandalf</code>" in response.data
    assert b"code>Gandalf the White</code>" in response.data
    assert b"code>Mithrandir</code>" in response.data

    # Test Gandalf at chapter 2 (should show v1 since no v2 exists yet)
    response = test_app_wiki.get("/wiki/1/gandalf")
    assert response.status_code == 200
    assert b"Gandalf the Grey" in response.data  # Should fall back to chapter 1 version
    assert b"long grey beard" in response.data


def test_wiki_page_not_found(test_app_wiki: FlaskClient) -> None:
    """Test 404 for non-existent wiki page."""
    response = test_app_wiki.get("/wiki/0/nonexistent-character")
    assert response.status_code == 404


def test_wiki_chapter_version_selection(
    test_app_wiki: FlaskClient,
    populated_wiki_db: SafeConnection,  # noqa: ARG001
) -> None:
    """Test that wiki pages show correct versions based on chapter."""
    # Bilbo exists in chapters 1 and 2

    # Chapter 1 - original version
    response = test_app_wiki.get("/wiki/0/bilbo-baggins")
    assert response.status_code == 200
    assert b"Bilbo Baggins" in response.data
    assert b"Former Ring-bearer" in response.data
    # Should not have the "Undying Lands" text from chapter 2
    assert b"Undying Lands" not in response.data

    # Chapter 2 - updated version
    response = test_app_wiki.get("/wiki/1/bilbo-baggins")
    assert response.status_code == 200
    assert b"Bilbo Baggins" in response.data
    assert b"Undying Lands" in response.data  # Chapter 2 addition

    # Chapter 3 - should show chapter 2 version (latest available)
    response = test_app_wiki.get("/wiki/2/bilbo-baggins")
    assert response.status_code == 200
    assert b"Undying Lands" in response.data  # Still chapter 2 version


def test_wiki_page_version_history(
    test_app_wiki: FlaskClient,
    populated_wiki_db: SafeConnection,  # noqa: ARG001
) -> None:
    """Test that wiki page has link to full history."""
    response = test_app_wiki.get("/wiki/2/gandalf")
    assert response.status_code == 200

    # Should have link to full history
    assert b"History" in response.data
    assert b'href="/history/2/gandalf"' in response.data

    # Should show current chapter info
    assert b"Chapter 3" in response.data


def test_wiki_page_metadata(
    test_app_wiki: FlaskClient,
    populated_wiki_db: SafeConnection,  # noqa: ARG001
) -> None:
    """Test that wiki page metadata is displayed correctly."""
    response = test_app_wiki.get("/wiki/1/frodo-baggins")
    assert response.status_code == 200

    # Should show page metadata
    assert b"frodo-baggins" in response.data  # slug
    assert b"Chapter 2" in response.data  # chapter
    assert b"Block #" in response.data  # source block link

    # Should show names/aliases
    assert b"Known as:" in response.data
    assert b"code>Frodo</code>" in response.data
    assert b"code>Frodo Baggins</code>" in response.data
    assert b"code>Mr. Frodo</code>" in response.data


def test_wiki_page_navigation(
    test_app_wiki: FlaskClient,
    populated_wiki_db: SafeConnection,  # noqa: ARG001
) -> None:
    """Test navigation elements on wiki pages."""
    response = test_app_wiki.get("/wiki/0/gandalf")
    assert response.status_code == 200

    # Should have link back to wiki list
    assert b'href="/wiki"' in response.data  # Link back to wiki list
    assert b"gandalf" in response.data  # Page content

    # Should have action buttons
    # Chapter navigation is now handled through dropdown in nav

    # Should have links to conversation blocks
    assert b'href="/conversation/' in response.data


def test_wiki_list_navigation(
    test_app_wiki: FlaskClient,
    populated_wiki_db: SafeConnection,  # noqa: ARG001
) -> None:
    """Test navigation from wiki list to individual pages."""
    response = test_app_wiki.get("/wiki")
    # Should redirect to latest started chapter
    assert response.status_code == 302

    # Follow the redirect to test navigation
    response = test_app_wiki.get("/wiki", follow_redirects=True)
    assert response.status_code == 200

    # Should have links to pages at the current chapter (3)
    assert b'href="/wiki/2/gandalf"' in response.data  # Gandalf at chapter 3
    assert b'href="/wiki/2/frodo-baggins"' in response.data  # Frodo at chapter 3
    assert b'href="/wiki/2/bilbo-baggins"' in response.data  # Bilbo at chapter 3

    # Pages are now directly linked without "View Page" buttons


def test_wiki_content_formatting(
    test_app_wiki: FlaskClient,
    populated_wiki_db: SafeConnection,  # noqa: ARG001
) -> None:
    """Test that wiki content is formatted correctly."""
    response = test_app_wiki.get("/wiki/2/gandalf")
    assert response.status_code == 200

    # Markdown rendering should create proper paragraph tags
    assert b"<p>" in response.data

    # Content should be properly escaped and formatted
    assert b"Gandalf has returned as Gandalf the White" in response.data
    assert b"He is now even more powerful" in response.data


def test_wiki_markdown_rendering_and_links(
    test_app_wiki: FlaskClient, temp_db: SafeConnection
) -> None:
    """Test that Markdown content is rendered and wiki links are transformed."""
    with temp_db.transaction_cursor() as cursor:
        # Create test chapters
        Chapter.add_chapter(cursor, 0, ["Chapter 1"], "First chapter")
        Chapter.add_chapter(cursor, 1, ["Chapter 2"], "Second chapter")

        # Create conversations and mark chapters as started
        conv_id = Conversation.create(cursor)
        chapter0 = Chapter.read_chapter(cursor, 0)
        chapter1 = Chapter.read_chapter(cursor, 1)
        assert chapter0 and chapter1
        chapter0.set_conversation_id(conv_id)
        chapter1.set_conversation_id(conv_id)

        # Create a block
        block = Block.create_text(
            cursor, conv_id.id, conv_id.current_generation, "user", "Create wiki pages"
        )
        block.mark_as_sent()

        # Create a wiki page with Markdown content including various elements and links
        markdown_content = """# Character Overview

**Aragorn** is a *ranger* from the North.

## Key Information
- Real name: Aragorn son of Arathorn
- Also known as: [Strider](strider)
- Home: [Rivendell](rivendell)
- Weapon: [Anduril](anduril)

He is connected to [Gandalf](gandalf-the-grey) and helps [Frodo](frodo-baggins).

### External References
- More info: [Tolkien Gateway](https://tolkiengateway.net)
- Full path link: [History Page](/history/1/aragorn)

## Recent Events
In recent chapters, he has been traveling with the Fellowship."""

        WikiPage.create(
            cursor,
            chapter_id=0,
            slug="aragorn",
            create_block_id=block.id,
            title="Aragorn",
            names=["Aragorn", "Strider"],
            summary="A ranger from the North",
            body=markdown_content,
        )

        # Also create the target pages for some links
        WikiPage.create(
            cursor,
            chapter_id=0,
            slug="strider",
            create_block_id=block.id,
            title="Strider",
            names=["Strider"],
            summary="Aragorn's alias",
            body="Strider is the name Aragorn uses in Bree.",
        )

        # Copy pages from chapter 0 to chapter 1 (proper chapter progression)
        WikiPage.copy_current_for_new_chapter(cursor, 1)

    # Test rendering at chapter 1 (index 1)
    response = test_app_wiki.get("/wiki/1/aragorn")
    assert response.status_code == 200

    # Test Markdown elements are rendered as HTML
    assert b"<h1>Character Overview</h1>" in response.data
    assert b"<h2>Key Information</h2>" in response.data
    assert b"<h3>External References</h3>" in response.data
    assert b"<strong>Aragorn</strong>" in response.data  # Bold
    assert b"<em>ranger</em>" in response.data  # Italic
    assert b"<ul>" in response.data and b"<li>" in response.data  # List

    # Test wiki links are transformed to chapter-aware URLs
    assert b'<a href="/wiki/1/strider">Strider</a>' in response.data
    assert b'<a href="/wiki/1/rivendell">Rivendell</a>' in response.data
    assert b'<a href="/wiki/1/anduril">Anduril</a>' in response.data
    assert b'<a href="/wiki/1/gandalf-the-grey">Gandalf</a>' in response.data
    assert b'<a href="/wiki/1/frodo-baggins">Frodo</a>' in response.data

    # Test external URLs get slug extracted (tolkiengateway.net from https://tolkiengateway.net)
    assert b'<a href="/wiki/1/tolkiengateway.net">Tolkien Gateway</a>' in response.data
    # Test absolute paths get slug extracted (aragorn from /history/1/aragorn)
    assert b'<a href="/wiki/1/aragorn">History Page</a>' in response.data


def test_wiki_single_name(test_app_wiki: FlaskClient, temp_db: SafeConnection) -> None:
    """Test wiki page display with only one name."""
    with temp_db.transaction_cursor() as cursor:
        # Add minimal data - chapter, conversation, block, wiki page (single name)
        Chapter.add_chapter(cursor, 0, ["Test"], "Test chapter")
        conv_id = Conversation.create(cursor)
        # Mark chapter as started
        chapter0 = Chapter.read_chapter(cursor, 0)
        assert chapter0 is not None
        chapter0.set_conversation_id(conv_id)
        block = Block.create_text(
            cursor, conv_id.id, conv_id.current_generation, "user", "Test"
        )
        block.mark_as_sent()

        WikiPage.create(
            cursor,
            chapter_id=0,
            slug="test-character",
            create_block_id=block.id,
            title="Test Character",
            names=["Test Character"],
            summary="A test character.",
            body="This is a test character with no aliases.",
        )

    response = test_app_wiki.get("/wiki/0/test-character")
    assert response.status_code == 200
    assert b"Test Character" in response.data
    assert b"A test character." in response.data
    # With only one name, it should still show "Known as:"
    assert b"Known as:" in response.data
    assert b"code>Test Character</code>" in response.data


# Search functionality tests (for future implementation)
def test_wiki_search_single_result(
    test_app_wiki: FlaskClient,
    populated_wiki_db: SafeConnection,  # noqa: ARG001
) -> None:
    """Test wiki search with a single result."""
    response = test_app_wiki.get("/search/2?names=Gandalf")  # Search in chapter 3
    assert response.status_code == 200
    assert b"gandalf" in response.data
    assert b"Gandalf the White" in response.data  # Should show latest version


def test_wiki_search_multiple_results(
    test_app_wiki: FlaskClient,
    populated_wiki_db: SafeConnection,  # noqa: ARG001
) -> None:
    """Test wiki search with multiple results."""
    # Search for "Baggins" should return both Frodo and Bilbo
    response = test_app_wiki.get("/search/2?names=Baggins")  # Search in chapter 3
    assert response.status_code == 200
    assert b"frodo-baggins" in response.data
    assert b"bilbo-baggins" in response.data
    assert b"Frodo Baggins" in response.data
    assert b"Bilbo Baggins" in response.data


def test_wiki_search_no_results(test_app_wiki: FlaskClient) -> None:
    """Test wiki search with no results."""
    response = test_app_wiki.get("/search?names=Nonexistent")  # Use redirect route
    # Should redirect to latest chapter search (404 if no chapters)
    assert response.status_code == 404


def test_search_redirect_preserves_query_params(
    test_app_wiki: FlaskClient,
    populated_wiki_db: SafeConnection,  # noqa: ARG001
) -> None:
    """Test that /search redirects preserve query parameters."""
    # Test that query parameters are preserved through redirect
    response = test_app_wiki.get("/search?names=Gandalf")
    assert response.status_code == 302  # Should redirect

    # Follow the redirect and verify the query is preserved
    response = test_app_wiki.get("/search?names=Gandalf", follow_redirects=True)
    assert response.status_code == 200
    assert b"Gandalf" in response.data  # Should show search results


def test_wiki_search_multiple_names(
    test_app_wiki: FlaskClient,
    populated_wiki_db: SafeConnection,  # noqa: ARG001
) -> None:
    """Test wiki search with comma-separated names."""
    response = test_app_wiki.get("/search/2?names=Gandalf,Frodo")  # Search in chapter 3
    assert response.status_code == 200
    assert b"gandalf" in response.data
    assert b"frodo-baggins" in response.data


def test_wiki_search_fuzzy_matching(
    test_app_wiki: FlaskClient,
    populated_wiki_db: SafeConnection,  # noqa: ARG001
) -> None:
    """Test wiki search with partial/fuzzy matching."""
    # Partial match should work
    response = test_app_wiki.get("/search/2?names=Mith")  # Search in chapter 3
    assert response.status_code == 200
    assert b"gandalf" in response.data  # Should match "Mithrandir"


def test_wiki_chapter_filtering(
    test_app_wiki: FlaskClient,
    populated_wiki_db: SafeConnection,  # noqa: ARG001
) -> None:
    """Test that wiki pages are properly filtered by chapter."""
    # Test chapter 1 - should only show pages created in or before chapter 1
    response = test_app_wiki.get("/wiki/0")
    assert response.status_code == 200
    # Should show gandalf and bilbo (both created in ch1) but NOT frodo (created in ch2)
    assert b"gandalf" in response.data
    assert b"bilbo-baggins" in response.data
    assert b"frodo-baggins" not in response.data

    # Test chapter 2 - should show pages created in or before chapter 2
    response = test_app_wiki.get("/wiki/1")
    assert response.status_code == 200
    # Should show gandalf (ch1), frodo (ch2), and bilbo (updated ch2, originally ch1)
    assert b"gandalf" in response.data
    assert b"frodo-baggins" in response.data
    assert b"bilbo-baggins" in response.data

    # Test chapter 3 - should show all pages (including gandalf's update in ch3)
    response = test_app_wiki.get("/wiki/2")
    assert response.status_code == 200
    # Should show all characters
    assert b"gandalf" in response.data
    assert b"frodo-baggins" in response.data
    assert b"bilbo-baggins" in response.data


# History functionality tests
def test_wiki_history_list(
    test_app_wiki: FlaskClient,
    populated_wiki_db: SafeConnection,  # noqa: ARG001
) -> None:
    """Test wiki page history listing."""
    response = test_app_wiki.get("/history/2/gandalf")
    assert response.status_code == 200
    # Check table structure exists with striped class
    assert b'<table class="striped' in response.data
    assert b"<th>Chapter</th>" in response.data
    assert b"<th>Create Time</th>" in response.data
    # Check content in table rows - formatted chapter names
    assert b"<td>Book One - Chapter 1</td>" in response.data  # Chapter 0
    assert b"<td>Book One - Chapter 3</td>" in response.data  # Chapter 2


def test_wiki_history_detail(
    test_app_wiki: FlaskClient,
    populated_wiki_db: SafeConnection,  # noqa: ARG001
) -> None:
    """Test specific historical version display."""
    # Get specific historical version by reading the page
    with populated_wiki_db.transaction_cursor() as cursor:
        gandalf_page = WikiPage.read_page_at(cursor, "gandalf", 0)
        assert gandalf_page is not None
        page_id = gandalf_page.id

    response = test_app_wiki.get(f"/pageid/0/{page_id}")
    assert response.status_code == 200
    assert b"Gandalf the Grey" in response.data  # Historical version title
    assert b"long grey beard" in response.data  # Historical content
    # Historical version indication has changed in new layout
    # Main content should be historical, not current
    # This text is only in the newer version
    assert b"battle with the Balrog" not in response.data


def test_wiki_history_not_found(test_app_wiki: FlaskClient) -> None:
    """Test 404 for non-existent history."""
    response = test_app_wiki.get("/history/0/nonexistent")
    assert response.status_code == 404


def test_wiki_history_filtered_by_chapter(
    test_app_wiki: FlaskClient,
    populated_wiki_db: SafeConnection,  # noqa: ARG001
) -> None:
    """Test that wiki page history is filtered by chapter."""
    # Get history for gandalf at chapter 0 (should only show chapter 0 version)
    response = test_app_wiki.get("/history/0/gandalf")
    assert response.status_code == 200
    assert b"<td>Book One - Chapter 1</td>" in response.data  # Chapter 0
    assert (
        b"<td>Book One - Chapter 3</td>" not in response.data
    )  # Chapter 2 should not be visible

    # Get history for gandalf at chapter 2 (should show both versions)
    response = test_app_wiki.get("/history/2/gandalf")
    assert response.status_code == 200
    assert b"<td>Book One - Chapter 1</td>" in response.data  # Chapter 0
    assert b"<td>Book One - Chapter 3</td>" in response.data  # Chapter 2


def test_search_form_chapter_context(
    test_app_wiki: FlaskClient,
    populated_wiki_db: SafeConnection,  # noqa: ARG001
) -> None:
    """Test that search form action respects current chapter context."""
    # When on a specific chapter's wiki list, search form should use that chapter
    response = test_app_wiki.get("/wiki/1")
    assert response.status_code == 200
    # Search form should target the specific chapter (chapter 2)
    assert b'action="/search/1"' in response.data

    # When on a specific wiki page, search form should use that chapter
    response = test_app_wiki.get("/wiki/0/gandalf")
    assert response.status_code == 200
    # Search form should target the specific chapter (chapter 1)
    assert b'action="/search/0"' in response.data

    # When on search results for specific chapter, should maintain that chapter
    response = test_app_wiki.get("/search/2?names=Gandalf")
    assert response.status_code == 200
    # Search form should still target that same chapter (chapter 3)
    assert b'action="/search/2"' in response.data


def test_chapter_selector_dropdown(
    test_app_wiki: FlaskClient,
    populated_wiki_db: SafeConnection,  # noqa: ARG001
) -> None:
    """Test that chapter selector shows proper chapter names in dropdown."""
    response = test_app_wiki.get("/wiki/1")
    assert response.status_code == 200

    # Should have a dropdown with chapter names
    assert b'<details class="dropdown">' in response.data

    # Current chapter should be shown in summary
    assert b"<summary>" in response.data
    assert b"Book One - Chapter 2" in response.data

    # Should show chapter names as links in the dropdown
    assert b'href="/wiki/0"' in response.data
    assert b"Book One - Chapter 1" in response.data
    assert b'href="/wiki/1"' in response.data
    assert b"Book One - Chapter 2" in response.data
    assert b'href="/wiki/2"' in response.data
    assert b"Book One - Chapter 3" in response.data


def test_chapter_selector_url_templates(
    test_app_wiki: FlaskClient,
    populated_wiki_db: SafeConnection,  # noqa: ARG001
) -> None:
    """Test that chapter selector dropdown has correct links for navigation."""
    # Wiki list should have links for chapter switching
    response = test_app_wiki.get("/wiki/1")
    assert response.status_code == 200
    assert b'href="/wiki/0"' in response.data
    assert b'href="/wiki/1"' in response.data
    assert b'href="/wiki/2"' in response.data

    # Wiki page should have links preserving the slug
    response = test_app_wiki.get("/wiki/0/gandalf")
    assert response.status_code == 200
    assert b'href="/wiki/0/gandalf"' in response.data
    assert b'href="/wiki/1/gandalf"' in response.data
    assert b'href="/wiki/2/gandalf"' in response.data

    # Search page should have links preserving query params
    response = test_app_wiki.get("/search/2?names=Gandalf")
    assert response.status_code == 200
    assert b'href="/search/2?names=Gandalf"' in response.data
    assert b'href="/search/0?names=Gandalf"' in response.data
    assert b'href="/search/1?names=Gandalf"' in response.data
