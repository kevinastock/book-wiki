"""Tests for bookwiki WikiPage model."""

import time

import pytest

from bookwiki.db import SafeConnection
from bookwiki.models import Block, Chapter, Conversation, WikiPage


def test_wiki_page_create_and_read(temp_db: SafeConnection) -> None:
    """Test creating and reading a wiki page."""
    with temp_db.transaction_cursor() as cursor:
        # Set up prerequisites
        Chapter.add_chapter(cursor, 1, ["Book 1", "Chapter 1"], "Chapter content...")
        conversation = Conversation.create(cursor)
        block = Block.create_text(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            role="assistant",
            text="Creating wiki page",
        )

        # Create a wiki page using the block method
        wiki_page = block.write_wiki_page(
            chapter_id=1,
            slug="rand-althor",
            title="Rand al'Thor",
            names=["Rand al'Thor", "Dragon Reborn", "Car'a'carn"],
            summary="The Dragon Reborn and main protagonist",
            body="Rand al'Thor is a ta'veren and the Dragon Reborn...",
        )

        # Verify the created wiki page
        assert wiki_page.chapter_id == 1
        assert wiki_page.slug == "rand-althor"
        assert wiki_page.title == "Rand al'Thor"
        # Names are deduplicated and sorted, so check set equality
        assert set(wiki_page.names) == {"Rand al'Thor", "Dragon Reborn", "Car'a'carn"}
        assert wiki_page.summary == "The Dragon Reborn and main protagonist"
        assert wiki_page.body == "Rand al'Thor is a ta'veren and the Dragon Reborn..."
        assert wiki_page.create_block_id == block.id
        assert wiki_page.id is not None

        # Read the page back
        retrieved_page = WikiPage.read_page_at(cursor, "rand-althor", 1)

        assert retrieved_page is not None
        assert retrieved_page.id == wiki_page.id
        assert retrieved_page.chapter_id == 1
        assert retrieved_page.slug == "rand-althor"
        assert retrieved_page.title == "Rand al'Thor"
        assert set(retrieved_page.names) == {
            "Rand al'Thor",
            "Dragon Reborn",
            "Car'a'carn",
        }
        assert retrieved_page.summary == "The Dragon Reborn and main protagonist"
        assert (
            retrieved_page.body == "Rand al'Thor is a ta'veren and the Dragon Reborn..."
        )


def test_wiki_page_read_nonexistent(temp_db: SafeConnection) -> None:
    """Test reading a wiki page that doesn't exist."""
    with temp_db.transaction_cursor() as cursor:
        # Set up chapter
        Chapter.add_chapter(cursor, 1, ["Book 1", "Chapter 1"], "Chapter content...")

        # Try to read non-existent page
        page = WikiPage.read_page_at(cursor, "nonexistent-slug", 1)
        assert page is None

        # Try to read from non-existent chapter
        page = WikiPage.read_page_at(cursor, "any-slug", 999)
        assert page is None


def test_wiki_page_versioning_by_chapter(temp_db: SafeConnection) -> None:
    """Test that wiki pages can have different versions for different chapters."""
    with temp_db.transaction_cursor() as cursor:
        # Set up chapters and conversation
        Chapter.add_chapter(cursor, 1, ["Book 1", "Chapter 1"], "Chapter 1 content...")
        Chapter.add_chapter(cursor, 2, ["Book 1", "Chapter 2"], "Chapter 2 content...")
        conversation = Conversation.create(cursor)

        block1 = Block.create_text(
            cursor,
            conversation.id,
            conversation.current_generation,
            "assistant",
            "Creating page v1",
        )
        block2 = Block.create_text(
            cursor,
            conversation.id,
            conversation.current_generation,
            "assistant",
            "Creating page v2",
        )

        # Create page for chapter 1
        block1.write_wiki_page(
            chapter_id=1,
            slug="character",
            title="Character Name",
            names=["Character Name"],
            summary="Early appearance summary",
            body="Initially appears as a minor character...",
        )

        # Create updated page for chapter 2 (same slug, different chapter)
        block2.write_wiki_page(
            chapter_id=2,
            slug="character",
            title="Character Name",
            names=["Character Name", "The Important One"],
            summary="Becomes more important",
            body="Revealed to be a major character with significant role...",
        )

        # Should be able to read both versions
        retrieved_v1 = WikiPage.read_page_at(cursor, "character", 1)
        retrieved_v2 = WikiPage.read_page_at(cursor, "character", 2)

        assert retrieved_v1 is not None
        assert retrieved_v2 is not None

        # They should be different pages
        assert retrieved_v1.id != retrieved_v2.id

        # Chapter 1 version should have limited info
        assert retrieved_v1.summary == "Early appearance summary"
        assert retrieved_v1.names == ["Character Name"]
        assert "minor character" in retrieved_v1.body

        # Chapter 2 version should have updated info
        assert retrieved_v2.summary == "Becomes more important"
        assert set(retrieved_v2.names) == {"Character Name", "The Important One"}
        assert "major character" in retrieved_v2.body


def test_wiki_page_latest_version_by_create_time(temp_db: SafeConnection) -> None:
    """Test that read_page_at returns the latest version for same chapter."""
    with temp_db.transaction_cursor() as cursor:
        # Set up prerequisites
        Chapter.add_chapter(cursor, 1, ["Book 1", "Chapter 1"], "Chapter content...")
        conversation = Conversation.create(cursor)

        block1 = Block.create_text(
            cursor,
            conversation.id,
            conversation.current_generation,
            "assistant",
            "First version",
        )
        block2 = Block.create_text(
            cursor,
            conversation.id,
            conversation.current_generation,
            "assistant",
            "Second version",
        )

        # Create first version
        block1.write_wiki_page(
            chapter_id=1,
            slug="evolving-character",
            title="Evolving Character",
            names=["Character"],
            summary="First understanding",
            body="Initial analysis...",
        )

        # Small delay to ensure different timestamps
        time.sleep(0.001)

        # Create second version (same chapter, same slug, later time)
        block2.write_wiki_page(
            chapter_id=1,
            slug="evolving-character",
            title="Evolving Character",
            names=["Character", "Real Name"],
            summary="Updated understanding",
            body="Corrected analysis with new information...",
        )

        # Should get the latest version
        retrieved_page = WikiPage.read_page_at(cursor, "evolving-character", 1)

        assert retrieved_page is not None
        assert retrieved_page.summary == "Updated understanding"
        assert "Corrected analysis" in retrieved_page.body
        assert set(retrieved_page.names) == {"Character", "Real Name"}
        assert retrieved_page.create_block_id == block2.id


def test_wiki_page_names_junction_table(temp_db: SafeConnection) -> None:
    """Test that the wiki_name and wiki_page_name junction table works correctly."""
    with temp_db.transaction_cursor() as cursor:
        # Set up prerequisites
        Chapter.add_chapter(cursor, 1, ["Book 1", "Chapter 1"], "Chapter content...")
        conversation = Conversation.create(cursor)

        # Create multiple pages with overlapping names
        block1 = Block.create_text(
            cursor,
            conversation.id,
            conversation.current_generation,
            "assistant",
            "Creating page 1",
        )
        block2 = Block.create_text(
            cursor,
            conversation.id,
            conversation.current_generation,
            "assistant",
            "Creating page 2",
        )

        # First character with multiple names
        block1.write_wiki_page(
            chapter_id=1,
            slug="character-one",
            title="Character One",
            names=["Character One", "The Chosen", "Hero"],
            summary="First character",
            body="First character description",
        )

        # Second character with some overlapping names
        block2.write_wiki_page(
            chapter_id=1,
            slug="character-two",
            title="Character Two",
            names=["Character Two", "The Chosen", "Villain"],  # "The Chosen" overlaps
            summary="Second character",
            body="Second character description",
        )

        # Verify both pages can be retrieved
        retrieved_page1 = WikiPage.read_page_at(cursor, "character-one", 1)
        retrieved_page2 = WikiPage.read_page_at(cursor, "character-two", 1)

        assert retrieved_page1 is not None
        assert retrieved_page2 is not None

        # Check that names are correctly associated
        assert set(retrieved_page1.names) == {"Character One", "The Chosen", "Hero"}
        assert set(retrieved_page2.names) == {"Character Two", "The Chosen", "Villain"}

        # Verify that "The Chosen" is properly shared between both pages
        # This implicitly tests deduplication - if both pages can access the name,
        # the underlying deduplication is working correctly
        assert "The Chosen" in retrieved_page1.names
        assert "The Chosen" in retrieved_page2.names


def test_wiki_page_empty_names_list(temp_db: SafeConnection) -> None:
    """Test that creating a wiki page with no names raises an error."""
    with temp_db.transaction_cursor() as cursor:
        # Set up prerequisites
        Chapter.add_chapter(cursor, 1, ["Book 1", "Chapter 1"], "Chapter content...")
        conversation = Conversation.create(cursor)
        block = Block.create_text(
            cursor,
            conversation.id,
            conversation.current_generation,
            "assistant",
            "Creating nameless page",
        )

        # Trying to create a page with empty names should raise ValueError
        with pytest.raises(ValueError, match="Wiki pages must have at least one name"):
            block.write_wiki_page(
                chapter_id=1,
                slug="nameless-entity",
                title="Unknown Entity",
                names=[],
                summary="An entity with no known names",
                body="This entity has not been named yet...",
            )


def test_wiki_page_unicode_content(temp_db: SafeConnection) -> None:
    """Test creating wiki pages with unicode content."""
    with temp_db.transaction_cursor() as cursor:
        # Set up prerequisites
        Chapter.add_chapter(cursor, 1, ["Книга", "Глава 1"], "Содержание главы...")
        conversation = Conversation.create(cursor)
        block = Block.create_text(
            cursor,
            conversation.id,
            conversation.current_generation,
            "assistant",
            "Creating unicode page",
        )

        # Create page with unicode content
        block.write_wiki_page(
            chapter_id=1,
            slug="unicode-character",
            title="Персонаж",
            names=["Персонаж", "Герой", "龙"],
            summary="Описание персонажа с эмодзи 🐉",
            body="Это персонаж из фэнтези романа... 中文字符也可以 🌟",
        )

        # Read back and verify unicode is preserved
        retrieved_page = WikiPage.read_page_at(cursor, "unicode-character", 1)

        assert retrieved_page is not None
        assert retrieved_page.title == "Персонаж"
        # All three names should be preserved as they're distinct Unicode
        assert set(retrieved_page.names) == {"Персонаж", "Герой", "龙"}
        assert "🐉" in retrieved_page.summary
        assert "中文字符也可以" in retrieved_page.body
        assert "🌟" in retrieved_page.body


def test_wiki_page_large_content(temp_db: SafeConnection) -> None:
    """Test creating wiki pages with large content."""
    with temp_db.transaction_cursor() as cursor:
        # Set up prerequisites
        Chapter.add_chapter(cursor, 1, ["Book 1", "Chapter 1"], "Chapter content...")
        conversation = Conversation.create(cursor)
        block = Block.create_text(
            cursor,
            conversation.id,
            conversation.current_generation,
            "assistant",
            "Creating large page",
        )

        # Create large content
        large_summary = "Very detailed character analysis. " * 100  # ~3KB
        large_body = (
            "This is a comprehensive character analysis that goes into great detail "
            "about every aspect of the character's personality, history, "
            "motivations, and relationships. " * 200
        )  # ~26KB

        block.write_wiki_page(
            chapter_id=1,
            slug="detailed-character",
            title="Very Detailed Character",
            names=["Detailed Character", "Complex Person", "Multi-faceted Individual"],
            summary=large_summary,
            body=large_body,
        )

        # Read back and verify large content is preserved
        retrieved_page = WikiPage.read_page_at(cursor, "detailed-character", 1)

        assert retrieved_page is not None
        assert len(retrieved_page.summary) > 3000
        assert len(retrieved_page.body) > 25000
        assert retrieved_page.summary == large_summary
        assert retrieved_page.body == large_body


def test_get_name_slug_pairs_basic(temp_db: SafeConnection) -> None:
    """Test getting name-slug pairs from multiple wiki pages."""
    with temp_db.transaction_cursor() as cursor:
        # Set up chapters with proper conversations
        chapter1 = Chapter.add_chapter(
            cursor, 1, ["Book 1", "Chapter 1"], "Chapter 1 content..."
        )
        Chapter.add_chapter(cursor, 2, ["Book 1", "Chapter 2"], "Chapter 2 content...")

        # Set up chapter 1 processing
        conv1 = Conversation.create(cursor)
        chapter1.set_conversation_id(conv1)

        # Create blocks for creating pages during chapter 1
        block1 = Block.create_text(
            cursor,
            conv1.id,
            conv1.current_generation,
            "assistant",
            "Creating page 1",
        )
        block2 = Block.create_text(
            cursor,
            conv1.id,
            conv1.current_generation,
            "assistant",
            "Creating page 2",
        )

        # Create pages during chapter 1 processing
        block1.write_wiki_page(
            chapter_id=1,
            slug="character-one",
            title="Character One",
            names=["Character One", "The Hero", "Chosen One"],
            summary="First character",
            body="Character one description",
        )

        block2.write_wiki_page(
            chapter_id=1,
            slug="character-two",
            title="Character Two",
            names=["Character Two", "The Villain"],
            summary="Second character",
            body="Character two description",
        )

        # Chapter 2 processing: inherit chapter 1 pages, then add new content
        WikiPage.copy_current_for_new_chapter(cursor, 2)

        chapter2 = Chapter.read_chapter(cursor, 2)
        assert chapter2 is not None
        conv2 = Conversation.create(cursor)
        chapter2.set_conversation_id(conv2)

        block3 = Block.create_text(
            cursor,
            conv2.id,
            conv2.current_generation,
            "assistant",
            "Creating page 3",
        )

        # Create new page in chapter 2
        block3.write_wiki_page(
            chapter_id=2,
            slug="character-three",
            title="Character Three",
            names=["Character Three", "The Mentor", "Wise One"],
            summary="Third character",
            body="Character three description",
        )

        # Test getting pairs for chapter 1 (should only include first two pages)
        pairs_ch1 = WikiPage.get_name_slug_pairs(cursor, 1)
        expected_ch1 = [
            ("Character One", "character-one"),
            ("Character Two", "character-two"),
            ("Chosen One", "character-one"),
            ("The Hero", "character-one"),
            ("The Villain", "character-two"),
        ]

        assert sorted(pairs_ch1) == sorted(expected_ch1)

        # Test getting pairs for chapter 2 (should include all three pages)
        pairs_ch2 = WikiPage.get_name_slug_pairs(cursor, 2)
        expected_ch2 = expected_ch1 + [
            ("Character Three", "character-three"),
            ("The Mentor", "character-three"),
            ("Wise One", "character-three"),
        ]

        assert sorted(pairs_ch2) == sorted(expected_ch2)


def test_get_name_slug_pairs_empty(temp_db: SafeConnection) -> None:
    """Test getting name-slug pairs when no wiki pages exist."""
    with temp_db.transaction_cursor() as cursor:
        # Set up a chapter but no wiki pages
        Chapter.add_chapter(cursor, 1, ["Book 1", "Chapter 1"], "Chapter 1 content...")

        # Should return empty list
        pairs = WikiPage.get_name_slug_pairs(cursor, 1)
        assert pairs == []


def test_get_name_slug_pairs_with_duplicates(temp_db: SafeConnection) -> None:
    """Test that duplicate names pointing to the same slug are handled correctly."""
    with temp_db.transaction_cursor() as cursor:
        # Set up chapter
        Chapter.add_chapter(cursor, 1, ["Book 1", "Chapter 1"], "Chapter 1 content...")
        conversation = Conversation.create(cursor)

        # Create two different versions of the same page in same chapter
        block1 = Block.create_text(
            cursor,
            conversation.id,
            conversation.current_generation,
            "assistant",
            "Creating page v1",
        )
        block2 = Block.create_text(
            cursor,
            conversation.id,
            conversation.current_generation,
            "assistant",
            "Creating page v2",
        )

        # First version with initial names
        block1.write_wiki_page(
            chapter_id=1,
            slug="evolving-character",
            title="Evolving Character",
            names=["Character", "Unknown Person"],
            summary="Initial version",
            body="First understanding",
        )

        # Second version with additional names (same chapter, same slug)
        block2.write_wiki_page(
            chapter_id=1,
            slug="evolving-character",
            title="Evolving Character",
            names=["Character", "Real Name", "True Identity"],
            summary="Updated version",
            body="Better understanding",
        )

        # Should get names from the latest version only (the second version)
        pairs = WikiPage.get_name_slug_pairs(cursor, 1)
        expected = [
            ("Character", "evolving-character"),
            ("Real Name", "evolving-character"),
            ("True Identity", "evolving-character"),
        ]

        assert sorted(pairs) == sorted(expected)


def test_get_name_slug_pairs_chapter_filtering(temp_db: SafeConnection) -> None:
    """Test that chapter filtering works correctly."""
    with temp_db.transaction_cursor() as cursor:
        # Set up chapters with proper conversations - simulating sequential processing
        chapter1 = Chapter.add_chapter(
            cursor, 1, ["Book 1", "Chapter 1"], "Chapter 1 content..."
        )
        Chapter.add_chapter(cursor, 2, ["Book 1", "Chapter 2"], "Chapter 2 content...")
        chapter3 = Chapter.add_chapter(
            cursor, 3, ["Book 1", "Chapter 3"], "Chapter 3 content..."
        )
        chapter4 = Chapter.add_chapter(
            cursor, 4, ["Book 1", "Chapter 4"], "Chapter 4 content..."
        )

        # Chapter 1 processing: create early character
        conv1 = Conversation.create(cursor)
        chapter1.set_conversation_id(conv1)

        block1 = Block.create_text(
            cursor,
            conv1.id,
            conv1.current_generation,
            "assistant",
            "Creating page 1",
        )

        block1.write_wiki_page(
            chapter_id=1,
            slug="early-character",
            title="Early Character",
            names=["Early Character"],
            summary="Appears early",
            body="Early appearance",
        )

        # Chapter 2 processing: inherit pages (no new content)
        WikiPage.copy_current_for_new_chapter(cursor, 2)

        # Chapter 3 processing: inherit pages, then add new content
        WikiPage.copy_current_for_new_chapter(cursor, 3)

        conv3 = Conversation.create(cursor)
        chapter3.set_conversation_id(conv3)

        block2 = Block.create_text(
            cursor,
            conv3.id,
            conv3.current_generation,
            "assistant",
            "Creating page 2",
        )

        block2.write_wiki_page(
            chapter_id=3,
            slug="mid-character",
            title="Mid Character",
            names=["Mid Character"],
            summary="Appears in middle",
            body="Mid appearance",
        )

        # Chapter 4 processing: inherit pages from chapter 3, then add new content
        WikiPage.copy_current_for_new_chapter(cursor, 4)

        conv4 = Conversation.create(cursor)
        chapter4.set_conversation_id(conv4)

        block3 = Block.create_text(
            cursor,
            conv4.id,
            conv4.current_generation,
            "assistant",
            "Creating page 3",
        )

        block3.write_wiki_page(
            chapter_id=4,
            slug="late-character",
            title="Late Character",
            names=["Late Character"],
            summary="Appears late",
            body="Late appearance",
        )

        # Test chapter filtering

        # Chapter 1: only early character
        pairs_ch1 = WikiPage.get_name_slug_pairs(cursor, 1)
        assert pairs_ch1 == [("Early Character", "early-character")]

        # Chapter 2: should include early character (inherited from chapter 1)
        pairs_ch2 = WikiPage.get_name_slug_pairs(cursor, 2)
        assert pairs_ch2 == [("Early Character", "early-character")]

        # Chapter 3: should include early and mid characters
        pairs_ch3 = WikiPage.get_name_slug_pairs(cursor, 3)
        expected_ch3 = [
            ("Early Character", "early-character"),
            ("Mid Character", "mid-character"),
        ]
        assert sorted(pairs_ch3) == sorted(expected_ch3)

        # Chapter 4: should include all characters
        pairs_ch4 = WikiPage.get_name_slug_pairs(cursor, 4)
        expected_ch4 = [
            ("Early Character", "early-character"),
            ("Mid Character", "mid-character"),
            ("Late Character", "late-character"),
        ]
        assert sorted(pairs_ch4) == sorted(expected_ch4)


def test_read_page_at_cross_chapter_behavior(temp_db: SafeConnection) -> None:
    """Test that read_page_at returns most recent page from most recent chapter."""
    with temp_db.transaction_cursor() as cursor:
        # Set up multiple chapters
        Chapter.add_chapter(cursor, 1, ["Book 1", "Chapter 1"], "Chapter 1 content...")
        Chapter.add_chapter(cursor, 2, ["Book 1", "Chapter 2"], "Chapter 2 content...")
        Chapter.add_chapter(cursor, 3, ["Book 1", "Chapter 3"], "Chapter 3 content...")
        Chapter.add_chapter(cursor, 4, ["Book 1", "Chapter 4"], "Chapter 4 content...")
        Chapter.add_chapter(cursor, 5, ["Book 1", "Chapter 5"], "Chapter 5 content...")
        Chapter.add_chapter(
            cursor, 10, ["Book 1", "Chapter 10"], "Chapter 10 content..."
        )
        conversation = Conversation.create(cursor)

        # Create blocks for pages
        block1 = Block.create_text(
            cursor,
            conversation.id,
            conversation.current_generation,
            "assistant",
            "Page in chapter 1",
        )
        block3 = Block.create_text(
            cursor,
            conversation.id,
            conversation.current_generation,
            "assistant",
            "Page in chapter 3",
        )
        block5 = Block.create_text(
            cursor,
            conversation.id,
            conversation.current_generation,
            "assistant",
            "Page in chapter 5",
        )

        # Create same character's page in chapter 1
        block1.write_wiki_page(
            chapter_id=1,
            slug="main-character",
            title="Main Character",
            names=["Main Character"],
            summary="Early introduction",
            body="Character appears as a simple farm boy...",
        )

        # Copy chapter 1's wiki pages to chapter 2 (simulating processing chapter 2)
        WikiPage.copy_current_for_new_chapter(cursor, 2)

        # Add delay to ensure different timestamps
        time.sleep(0.001)

        # Update character's page in chapter 3
        block3.write_wiki_page(
            chapter_id=3,
            slug="main-character",
            title="Main Character",
            names=["Main Character", "The Chosen One"],
            summary="Powers revealed",
            body="Character discovers they have magical powers...",
        )

        # Copy chapter 3's wiki pages to chapter 4
        WikiPage.copy_current_for_new_chapter(cursor, 4)

        # Add delay to ensure different timestamps
        time.sleep(0.001)

        # Further update character's page in chapter 5
        block5.write_wiki_page(
            chapter_id=5,
            slug="main-character",
            title="Main Character",
            names=["Main Character", "The Chosen One", "Dragon Reborn"],
            summary="True identity revealed",
            body="Character is revealed to be the prophesied Dragon Reborn...",
        )

        # Simulate chapters 6-9 being processed without wiki changes
        for chapter_id in range(6, 10):
            WikiPage.copy_current_for_new_chapter(cursor, chapter_id)

        # Copy to chapter 10
        WikiPage.copy_current_for_new_chapter(cursor, 10)

        # Test reading at different chapters

        # Reading at chapter 1: should get chapter 1 version
        page_at_ch1 = WikiPage.read_page_at(cursor, "main-character", 1)
        assert page_at_ch1 is not None
        assert page_at_ch1.chapter_id == 1
        assert page_at_ch1.summary == "Early introduction"
        assert "farm boy" in page_at_ch1.body
        assert page_at_ch1.names == ["Main Character"]

        # Reading at chapter 2: should get chapter 1 version (most recent ≤ 2)
        page_at_ch2 = WikiPage.read_page_at(cursor, "main-character", 2)
        assert page_at_ch2 is not None
        assert page_at_ch2.chapter_id == 1
        assert page_at_ch2.summary == "Early introduction"
        assert "farm boy" in page_at_ch2.body

        # Reading at chapter 3: should get chapter 3 version
        page_at_ch3 = WikiPage.read_page_at(cursor, "main-character", 3)
        assert page_at_ch3 is not None
        assert page_at_ch3.chapter_id == 3
        assert page_at_ch3.summary == "Powers revealed"
        assert "magical powers" in page_at_ch3.body
        assert set(page_at_ch3.names) == {"Main Character", "The Chosen One"}

        # Reading at chapter 4: should get chapter 3 version (most recent ≤ 4)
        page_at_ch4 = WikiPage.read_page_at(cursor, "main-character", 4)
        assert page_at_ch4 is not None
        assert page_at_ch4.chapter_id == 3
        assert page_at_ch4.summary == "Powers revealed"

        # Reading at chapter 5: should get chapter 5 version
        page_at_ch5 = WikiPage.read_page_at(cursor, "main-character", 5)
        assert page_at_ch5 is not None
        assert page_at_ch5.chapter_id == 5
        assert page_at_ch5.summary == "True identity revealed"
        assert "Dragon Reborn" in page_at_ch5.body
        assert set(page_at_ch5.names) == {
            "Main Character",
            "The Chosen One",
            "Dragon Reborn",
        }

        # Reading at chapter 10: should get chapter 5 version (most recent ≤ 10)
        page_at_ch10 = WikiPage.read_page_at(cursor, "main-character", 10)
        assert page_at_ch10 is not None
        assert page_at_ch10.chapter_id == 5
        assert page_at_ch10.summary == "True identity revealed"


def test_read_page_at_multiple_pages_same_chapter_by_time(
    temp_db: SafeConnection,
) -> None:
    """Test that read_page_at returns most recent page by create_time in chapter."""
    with temp_db.transaction_cursor() as cursor:
        # Set up chapter
        Chapter.add_chapter(cursor, 2, ["Book 1", "Chapter 2"], "Chapter 2 content...")
        conversation = Conversation.create(cursor)

        # Create blocks for multiple revisions in same chapter
        block1 = Block.create_text(
            cursor,
            conversation.id,
            conversation.current_generation,
            "assistant",
            "First revision",
        )
        block2 = Block.create_text(
            cursor,
            conversation.id,
            conversation.current_generation,
            "assistant",
            "Second revision",
        )
        block3 = Block.create_text(
            cursor,
            conversation.id,
            conversation.current_generation,
            "assistant",
            "Third revision",
        )

        # First revision
        block1.write_wiki_page(
            chapter_id=2,
            slug="evolving-character",
            title="Evolving Character",
            names=["Character"],
            summary="First understanding",
            body="Initial analysis based on first appearance...",
        )

        # Small delay to ensure different timestamps
        time.sleep(0.001)

        # Second revision (same chapter, same slug)
        block2.write_wiki_page(
            chapter_id=2,
            slug="evolving-character",
            title="Evolving Character",
            names=["Character", "Hidden Identity"],
            summary="Updated understanding",
            body="Revised analysis after learning more about the character...",
        )

        # Small delay to ensure different timestamps
        time.sleep(0.001)

        # Third revision (same chapter, same slug)
        block3.write_wiki_page(
            chapter_id=2,
            slug="evolving-character",
            title="Evolving Character",
            names=["Character", "Hidden Identity", "True Name"],
            summary="Complete understanding",
            body="Final analysis with full knowledge of character's identity...",
        )

        # Should get the most recent version (third revision)
        retrieved_page = WikiPage.read_page_at(cursor, "evolving-character", 2)

        assert retrieved_page is not None
        assert retrieved_page.chapter_id == 2
        assert retrieved_page.summary == "Complete understanding"
        assert "Final analysis" in retrieved_page.body
        assert set(retrieved_page.names) == {
            "Character",
            "Hidden Identity",
            "True Name",
        }
        assert retrieved_page.create_block_id == block3.id


def test_read_page_at_prioritize_chapter_over_time(temp_db: SafeConnection) -> None:
    """Test that read_page_at prioritizes higher chapters over older timestamps."""
    with temp_db.transaction_cursor() as cursor:
        # Set up chapters
        Chapter.add_chapter(cursor, 2, ["Book 1", "Chapter 2"], "Chapter 2 content...")
        Chapter.add_chapter(cursor, 3, ["Book 1", "Chapter 3"], "Chapter 3 content...")
        Chapter.add_chapter(cursor, 4, ["Book 1", "Chapter 4"], "Chapter 4 content...")
        conversation = Conversation.create(cursor)

        # Create blocks
        block_ch2 = Block.create_text(
            cursor,
            conversation.id,
            conversation.current_generation,
            "assistant",
            "Chapter 2 page",
        )
        block_ch4 = Block.create_text(
            cursor,
            conversation.id,
            conversation.current_generation,
            "assistant",
            "Chapter 4 page",
        )

        # Create page in chapter 4 FIRST (earlier timestamp)
        block_ch4.write_wiki_page(
            chapter_id=4,
            slug="test-character",
            title="Test Character",
            names=["Test Character"],
            summary="Later chapter info",
            body="Information available in chapter 4...",
        )

        # Small delay to ensure different timestamps
        time.sleep(0.001)

        # Create page in chapter 2 AFTER (later timestamp, but earlier chapter)
        block_ch2.write_wiki_page(
            chapter_id=2,
            slug="test-character",
            title="Test Character",
            names=["Test Character", "Early Name"],
            summary="Earlier chapter info",
            body="Information available in chapter 2...",
        )

        # Copy chapter 2's wiki pages to chapter 3 (simulating processing)
        WikiPage.copy_current_for_new_chapter(cursor, 3)

        # Reading at chapter 4: should get chapter 4 version despite earlier timestamp
        retrieved_page = WikiPage.read_page_at(cursor, "test-character", 4)

        assert retrieved_page is not None
        assert retrieved_page.chapter_id == 4
        assert retrieved_page.summary == "Later chapter info"
        assert "chapter 4" in retrieved_page.body
        assert retrieved_page.create_block_id == block_ch4.id

        # When reading at chapter 3, should get chapter 2 version (most recent ≤ 3)
        retrieved_page_ch3 = WikiPage.read_page_at(cursor, "test-character", 3)

        assert retrieved_page_ch3 is not None
        assert retrieved_page_ch3.chapter_id == 2
        assert retrieved_page_ch3.summary == "Earlier chapter info"
        assert "chapter 2" in retrieved_page_ch3.body


def test_get_all_pages_chapter_unique_slugs(temp_db: SafeConnection) -> None:
    """Test that get_all_pages_chapter returns only one page per slug."""
    with temp_db.transaction_cursor() as cursor:
        # Set up chapters
        chapter1 = Chapter.add_chapter(cursor, 1, ["Book", "Chapter 1"], "Content 1")
        chapter2 = Chapter.add_chapter(cursor, 2, ["Book", "Chapter 2"], "Content 2")
        chapter3 = Chapter.add_chapter(cursor, 3, ["Book", "Chapter 3"], "Content 3")

        # Create conversations for each chapter (simulating the processor)
        conv1 = Conversation.create(cursor)
        conv2 = Conversation.create(cursor)
        conv3 = Conversation.create(cursor)

        # Start processing chapter 1
        chapter1.set_conversation_id(conv1)

        # Create pages during chapter 1 processing
        block1a = conv1.add_assistant_text("Creating character A")
        block1b = conv1.add_assistant_text("Creating character B")

        WikiPage.create(
            cursor,
            chapter_id=1,
            slug="character-a",
            create_block_id=block1a.id,
            title="Character A",
            names=["Alice", "Character A"],
            summary="Early appearance",
            body="Character A appears early...",
        )

        WikiPage.create(
            cursor,
            chapter_id=1,
            slug="character-b",
            create_block_id=block1b.id,
            title="Character B",
            names=["Bob", "Character B"],
            summary="Supporting character",
            body="Character B provides support...",
        )

        # Add delay to ensure different create_time
        time.sleep(0.001)

        # Start processing chapter 2
        chapter2.set_conversation_id(conv2)

        # Inherit pages from chapter 1
        WikiPage.copy_current_for_new_chapter(cursor, 2)

        # Create/update pages during chapter 2 processing
        block2a = conv2.add_assistant_text("Updating character A")
        block2b = conv2.add_assistant_text("Updating character B")
        block2c = conv2.add_assistant_text("Creating character C")

        WikiPage.create(
            cursor,
            chapter_id=2,
            slug="character-a",
            create_block_id=block2a.id,
            title="Character A Updated",
            names=["Alice", "Character A", "The Hero"],
            summary="Character development",
            body="Character A grows in importance...",
        )

        WikiPage.create(
            cursor,
            chapter_id=2,
            slug="character-b",
            create_block_id=block2b.id,
            title="Character B Evolved",
            names=["Bob", "Character B", "The Mentor"],
            summary="Mentor role revealed",
            body="Character B becomes a mentor...",
        )

        WikiPage.create(
            cursor,
            chapter_id=2,
            slug="character-c",
            create_block_id=block2c.id,
            title="Character C",
            names=["Charlie", "Character C"],
            summary="New arrival",
            body="Character C joins the story...",
        )

        # Add delay to ensure different create_time
        time.sleep(0.001)

        # Start processing chapter 3
        chapter3.set_conversation_id(conv3)

        # Inherit pages from chapter 2
        WikiPage.copy_current_for_new_chapter(cursor, 3)

        # Update character A during chapter 3 processing
        block3a = conv3.add_assistant_text("Final update to character A")

        WikiPage.create(
            cursor,
            chapter_id=3,
            slug="character-a",
            create_block_id=block3a.id,
            title="Character A Final",
            names=["Alice", "Character A", "The Hero", "The Chosen One"],
            summary="Final form",
            body="Character A reaches their full potential...",
        )

        # Test getting all pages at chapter 3 (should get latest version of each slug)
        pages = WikiPage.get_all_pages_chapter(cursor, 3)

        # Should have exactly 3 pages (one for each unique slug)
        assert len(pages) == 3

        # Get slugs from results
        slugs = [page.slug for page in pages]

        # Should have exactly one of each slug
        assert "character-a" in slugs
        assert "character-b" in slugs
        assert "character-c" in slugs

        # Verify no duplicate slugs
        assert len(slugs) == len(set(slugs))

        # Verify we got the latest versions
        for page in pages:
            if page.slug == "character-a":
                assert page.chapter_id == 3
                assert page.summary == "Final form"
                assert "full potential" in page.body
                assert set(page.names) == {
                    "Alice",
                    "Character A",
                    "The Hero",
                    "The Chosen One",
                }
            elif page.slug == "character-b":
                assert page.chapter_id == 2
                assert page.summary == "Mentor role revealed"
                assert "mentor" in page.body
                assert set(page.names) == {"Bob", "Character B", "The Mentor"}
            elif page.slug == "character-c":
                assert page.chapter_id == 2
                assert page.summary == "New arrival"
                assert "joins the story" in page.body
                assert set(page.names) == {"Charlie", "Character C"}

        # Test at chapter 1 (should only get character-a and character-b, v1 each)
        pages_ch1 = WikiPage.get_all_pages_chapter(cursor, 1)
        assert len(pages_ch1) == 2

        slugs_ch1 = [page.slug for page in pages_ch1]
        assert "character-a" in slugs_ch1
        assert "character-b" in slugs_ch1
        assert "character-c" not in slugs_ch1  # Doesn't exist until chapter 2

        # Verify no duplicate slugs at chapter 1
        assert len(slugs_ch1) == len(set(slugs_ch1))

        # Verify we got chapter 1 versions
        for page in pages_ch1:
            assert page.chapter_id == 1
            if page.slug == "character-a":
                assert page.summary == "Early appearance"
            elif page.slug == "character-b":
                assert page.summary == "Supporting character"


def test_get_all_pages_chapter_same_chapter_multiple_times(
    temp_db: SafeConnection,
) -> None:
    """Test get_all_pages_chapter with multiple versions in the same chapter."""
    with temp_db.transaction_cursor() as cursor:
        conversation = Conversation.create(cursor)

        # Create blocks for multiple revisions
        block1 = Block.create_text(
            cursor,
            conversation.id,
            conversation.current_generation,
            "assistant",
            "First version",
        )
        block2 = Block.create_text(
            cursor,
            conversation.id,
            conversation.current_generation,
            "assistant",
            "Second version",
        )
        block3 = Block.create_text(
            cursor,
            conversation.id,
            conversation.current_generation,
            "assistant",
            "Third version",
        )

        # Create first version
        WikiPage.create(
            cursor,
            chapter_id=2,
            slug="evolving-char",
            create_block_id=block1.id,
            title="Evolving Character",
            names=["Character"],
            summary="First understanding",
            body="Initial analysis...",
        )

        # Add delay to ensure different create_time
        time.sleep(0.001)

        # Create second version (same chapter, same slug)
        WikiPage.create(
            cursor,
            chapter_id=2,
            slug="evolving-char",
            create_block_id=block2.id,
            title="Evolving Character",
            names=["Character", "Hidden Name"],
            summary="Better understanding",
            body="More detailed analysis...",
        )

        # Add delay to ensure different create_time
        time.sleep(0.001)

        # Create third version (same chapter, same slug)
        WikiPage.create(
            cursor,
            chapter_id=2,
            slug="evolving-char",
            create_block_id=block3.id,
            title="Evolving Character",
            names=["Character", "Hidden Name", "True Identity"],
            summary="Complete understanding",
            body="Final analysis with full knowledge...",
        )

        # Should get only the latest version (by create_time)
        pages = WikiPage.get_all_pages_chapter(cursor, 2)

        assert len(pages) == 1
        page = pages[0]

        assert page.slug == "evolving-char"
        assert page.chapter_id == 2
        assert page.summary == "Complete understanding"
        assert "Final analysis" in page.body
        assert set(page.names) == {"Character", "Hidden Name", "True Identity"}
        assert page.create_block_id == block3.id


def test_get_all_pages_chapter_empty_results(temp_db: SafeConnection) -> None:
    """Test get_all_pages_chapter when no pages exist at or before chapter."""
    with temp_db.transaction_cursor() as cursor:
        conversation = Conversation.create(cursor)
        block = Block.create_text(
            cursor,
            conversation.id,
            conversation.current_generation,
            "assistant",
            "Late page",
        )

        # Create a page in chapter 5
        WikiPage.create(
            cursor,
            chapter_id=5,
            slug="late-character",
            create_block_id=block.id,
            title="Late Character",
            names=["Late Character"],
            summary="Appears later",
            body="This character doesn't appear until chapter 5...",
        )

        # Requesting pages at chapter 3 should return empty (no pages ≤ 3)
        pages = WikiPage.get_all_pages_chapter(cursor, 3)
        assert pages == []

        # Requesting pages at chapter 1 should return empty
        pages_ch1 = WikiPage.get_all_pages_chapter(cursor, 1)
        assert pages_ch1 == []

        # Requesting pages at chapter 5 should return the page
        pages_ch5 = WikiPage.get_all_pages_chapter(cursor, 5)
        assert len(pages_ch5) == 1
        assert pages_ch5[0].slug == "late-character"


def test_get_all_pages_chapter_ordering_by_title(temp_db: SafeConnection) -> None:
    """Test that get_all_pages_chapter returns pages ordered by title."""
    with temp_db.transaction_cursor() as cursor:
        conversation = Conversation.create(cursor)
        block = Block.create_text(
            cursor,
            conversation.id,
            conversation.current_generation,
            "assistant",
            "Create pages",
        )

        # Create pages with titles that should be ordered alphabetically
        WikiPage.create(
            cursor,
            chapter_id=1,
            slug="zebra-char",
            create_block_id=block.id,
            title="Zebra Character",
            names=["Zebra", "Last One"],
            summary="Should be last alphabetically",
            body="This character has a name starting with Z...",
        )

        WikiPage.create(
            cursor,
            chapter_id=1,
            slug="alpha-char",
            create_block_id=block.id,
            title="Alpha Character",
            names=["Alpha", "First One"],
            summary="Should be first alphabetically",
            body="This character has a name starting with A...",
        )

        WikiPage.create(
            cursor,
            chapter_id=1,
            slug="beta-char",
            create_block_id=block.id,
            title="Beta Character",
            names=["Beta", "Middle One"],
            summary="Should be in the middle",
            body="This character has a name starting with B...",
        )

        # Get all pages
        pages = WikiPage.get_all_pages_chapter(cursor, 1)

        # Should be ordered by title alphabetically
        assert len(pages) == 3
        assert pages[0].slug == "alpha-char"  # "Alpha Character" comes first
        assert pages[1].slug == "beta-char"  # "Beta Character" comes second
        assert pages[2].slug == "zebra-char"  # "Zebra Character" comes last

        # Verify the names are correctly associated
        assert "Alpha" in pages[0].names
        assert "Beta" in pages[1].names
        assert "Zebra" in pages[2].names


def test_get_all_pages_chapter_sequential_progression(temp_db: SafeConnection) -> None:
    """Test get_all_pages_chapter with sequential chapter progression."""
    with temp_db.transaction_cursor() as cursor:
        # Set up chapters sequentially
        chapter1 = Chapter.add_chapter(
            cursor, 1, ["Book 1", "Chapter 1"], "Chapter 1 content..."
        )
        Chapter.add_chapter(cursor, 2, ["Book 1", "Chapter 2"], "Chapter 2 content...")
        chapter3 = Chapter.add_chapter(
            cursor, 3, ["Book 1", "Chapter 3"], "Chapter 3 content..."
        )
        Chapter.add_chapter(cursor, 4, ["Book 1", "Chapter 4"], "Chapter 4 content...")
        chapter5 = Chapter.add_chapter(
            cursor, 5, ["Book 1", "Chapter 5"], "Chapter 5 content..."
        )

        # Chapter 1 processing: create early character
        conv1 = Conversation.create(cursor)
        chapter1.set_conversation_id(conv1)

        block1 = Block.create_text(
            cursor,
            conv1.id,
            conv1.current_generation,
            "assistant",
            "Chapter 1",
        )

        WikiPage.create(
            cursor,
            chapter_id=1,
            slug="early-char",
            create_block_id=block1.id,
            title="Early Character",
            names=["Early"],
            summary="Appears in chapter 1",
            body="Very early appearance...",
        )

        # Chapter 2 processing: inherit pages from chapter 1 (no new content)
        WikiPage.copy_current_for_new_chapter(cursor, 2)

        # Chapter 3 processing: inherit pages, add new character and update existing
        WikiPage.copy_current_for_new_chapter(cursor, 3)

        conv3 = Conversation.create(cursor)
        chapter3.set_conversation_id(conv3)

        block3_new = Block.create_text(
            cursor,
            conv3.id,
            conv3.current_generation,
            "assistant",
            "Chapter 3 new character",
        )
        block3_update = Block.create_text(
            cursor,
            conv3.id,
            conv3.current_generation,
            "assistant",
            "Chapter 3 character update",
        )

        # Add delay to ensure different create_time
        time.sleep(0.001)

        WikiPage.create(
            cursor,
            chapter_id=3,
            slug="mid-char",
            create_block_id=block3_new.id,
            title="Mid Character",
            names=["Mid"],
            summary="Appears in chapter 3",
            body="Appears in the middle...",
        )

        # Add delay to ensure different create_time
        time.sleep(0.001)

        # Update early character in chapter 3
        WikiPage.create(
            cursor,
            chapter_id=3,
            slug="early-char",
            create_block_id=block3_update.id,
            title="Early Character Updated",
            names=["Early", "Evolved"],
            summary="Character development",
            body="Character has evolved by chapter 3...",
        )

        # Chapter 4 processing: inherit pages from chapter 3
        WikiPage.copy_current_for_new_chapter(cursor, 4)

        # Chapter 5 processing: inherit pages, add late character
        WikiPage.copy_current_for_new_chapter(cursor, 5)

        conv5 = Conversation.create(cursor)
        chapter5.set_conversation_id(conv5)

        block5 = Block.create_text(
            cursor,
            conv5.id,
            conv5.current_generation,
            "assistant",
            "Chapter 5",
        )

        WikiPage.create(
            cursor,
            chapter_id=5,
            slug="late-char",
            create_block_id=block5.id,
            title="Late Character",
            names=["Late"],
            summary="Appears in chapter 5",
            body="Very late appearance...",
        )

        # Test different chapter queries

        # Chapter 1: only early character (original version)
        pages_ch1 = WikiPage.get_all_pages_chapter(cursor, 1)
        assert len(pages_ch1) == 1
        assert pages_ch1[0].slug == "early-char"
        assert pages_ch1[0].chapter_id == 1
        assert pages_ch1[0].summary == "Appears in chapter 1"

        # Chapter 2: should include early character (inherited, original version)
        pages_ch2 = WikiPage.get_all_pages_chapter(cursor, 2)
        assert len(pages_ch2) == 1
        assert pages_ch2[0].slug == "early-char"
        assert (
            pages_ch2[0].chapter_id == 1
        )  # Still chapter 1 version since no updates until ch 3

        # Chapter 3: should include early (updated) and mid characters
        pages_ch3 = WikiPage.get_all_pages_chapter(cursor, 3)
        assert len(pages_ch3) == 2

        slugs_ch3 = {page.slug for page in pages_ch3}
        assert slugs_ch3 == {"early-char", "mid-char"}

        # Find the early character page - should be updated version
        early_page = next(p for p in pages_ch3 if p.slug == "early-char")
        assert early_page.chapter_id == 3
        assert early_page.summary == "Character development"
        assert set(early_page.names) == {"Early", "Evolved"}

        # Chapter 4: should still have early (ch3) and mid (ch3) characters
        pages_ch4 = WikiPage.get_all_pages_chapter(cursor, 4)
        assert len(pages_ch4) == 2
        slugs_ch4 = {page.slug for page in pages_ch4}
        assert slugs_ch4 == {"early-char", "mid-char"}

        # Chapter 5: should include all three characters
        pages_ch5 = WikiPage.get_all_pages_chapter(cursor, 5)
        assert len(pages_ch5) == 3
        slugs_ch5 = {page.slug for page in pages_ch5}
        assert slugs_ch5 == {"early-char", "mid-char", "late-char"}


def test_get_all_pages_chapter_rrf_body_length_tiebreaker(
    temp_db: SafeConnection,
) -> None:
    """Test get_all_pages_chapter RRF ordering with body length tiebreaker."""
    with temp_db.transaction_cursor() as cursor:
        conversation = Conversation.create(cursor)
        block = Block.create_text(
            cursor,
            conversation.id,
            conversation.current_generation,
            "assistant",
            "Create pages",
        )

        # Create pages with same chapter_id and distinct_chapters (all tie in RRF)
        # but different body lengths to test tiebreaker
        WikiPage.create(
            cursor,
            chapter_id=1,
            slug="character-1",
            create_block_id=block.id,
            title="Charlie's Story",
            names=["Alpha", "First Name"],
            summary="First character",
            body="Short",  # Shortest body
        )

        WikiPage.create(
            cursor,
            chapter_id=1,
            slug="character-2",
            create_block_id=block.id,
            title="Alpha Adventures",
            names=["Zebra", "Last Name"],
            summary="Second character",
            body="Medium length body",  # Medium body
        )

        WikiPage.create(
            cursor,
            chapter_id=1,
            slug="character-3",
            create_block_id=block.id,
            title="Beta's Tale",
            names=["Middle", "Name"],
            summary="Third character",
            body="This is the longest body content of all three pages",  # Longest body
        )

        # Get pages - ordered by body length when RRF scores tie
        pages = WikiPage.get_all_pages_chapter(cursor, 1)

        assert len(pages) == 3
        # All pages have same chapter_id=1 and distinct_chapters=1, so RRF scores tie
        # Tiebreaker is body length (longer first)
        assert pages[0].slug == "character-3"  # Longest body comes first
        assert pages[1].slug == "character-2"  # Medium body comes second
        assert pages[2].slug == "character-1"  # Shortest body comes last
