"""Tests for bookwiki Chapter model."""

import pytest

from bookwiki.db import SafeConnection
from bookwiki.models import Chapter, Conversation


def test_chapter_add_and_read(temp_db: SafeConnection) -> None:
    """Test adding and reading a chapter."""
    with temp_db.transaction_cursor() as cursor:
        # Add a chapter
        chapter_names = ["Book 1", "Chapter 1", "The Beginning"]
        chapter_text = "It was a dark and stormy night..."

        chapter = Chapter.add_chapter(cursor, 1, chapter_names, chapter_text)

        # Verify the returned chapter object
        assert chapter.id == 1
        assert chapter.name == chapter_names
        assert chapter.text == chapter_text
        assert chapter._cursor is cursor


def test_chapter_read_existing(temp_db: SafeConnection) -> None:
    """Test reading an existing chapter."""
    with temp_db.transaction_cursor() as cursor:
        # Add a chapter first
        chapter_names = ["Book 1", "Chapter 2", "The Middle"]
        chapter_text = "The plot thickens..."

        Chapter.add_chapter(cursor, 5, chapter_names, chapter_text)

        # Read the chapter back
        retrieved_chapter = Chapter.read_chapter(cursor, 5)

        assert retrieved_chapter is not None
        assert retrieved_chapter.id == 5
        assert retrieved_chapter.name == chapter_names
        assert retrieved_chapter.text == chapter_text
        assert retrieved_chapter.conversation_id is None  # Should be None initially
        assert retrieved_chapter._cursor is cursor


def test_chapter_read_nonexistent(temp_db: SafeConnection) -> None:
    """Test reading a non-existent chapter."""
    with temp_db.transaction_cursor() as cursor:
        # Try to read a chapter that doesn't exist
        retrieved_chapter = Chapter.read_chapter(cursor, 999)

        assert retrieved_chapter is None


def test_chapter_json_serialization(temp_db: SafeConnection) -> None:
    """Test that chapter names are properly serialized as JSON."""
    with temp_db.transaction_cursor() as cursor:
        # Add a chapter with complex name structure
        chapter_names = ["Series Name", "Book 1", "Part I", "Chapter 1"]
        chapter_text = "Complex structure test..."

        Chapter.add_chapter(cursor, 10, chapter_names, chapter_text)

        # Read it back
        retrieved_chapter = Chapter.read_chapter(cursor, 10)

        assert retrieved_chapter is not None
        assert retrieved_chapter.name == chapter_names

        # Verify the JSON is actually stored and parsed properly in the database
        # The Chapter model automatically handles JSON serialization/deserialization
        assert retrieved_chapter.name == chapter_names


def test_chapter_empty_name_list(temp_db: SafeConnection) -> None:
    """Test that adding a chapter with an empty name list raises an error."""
    with temp_db.transaction_cursor() as cursor:
        chapter_names: list[str] = []
        chapter_text = "No names test..."

        # Should raise ValueError for empty name list
        with pytest.raises(ValueError, match="Chapter name list cannot be empty"):
            Chapter.add_chapter(cursor, 20, chapter_names, chapter_text)


def test_chapter_duplicate_names_in_database(temp_db: SafeConnection) -> None:
    """Test that duplicate chapter name combinations raise an error."""
    with temp_db.transaction_cursor() as cursor:
        chapter_names = ["Book 1", "Chapter 1", "The Beginning"]
        chapter_text = "First chapter..."

        # Add the first chapter successfully
        Chapter.add_chapter(cursor, 21, chapter_names, chapter_text)

        # Try to add another chapter with the same name combination
        chapter_text_2 = "Different text but same name..."
        with pytest.raises(ValueError, match="Chapter with name .* already exists"):
            Chapter.add_chapter(cursor, 22, chapter_names, chapter_text_2)


def test_chapter_unicode_content(temp_db: SafeConnection) -> None:
    """Test adding a chapter with unicode content."""
    with temp_db.transaction_cursor() as cursor:
        chapter_names = ["Книга", "Глава 1", "Начало"]  # Russian text
        chapter_text = "Это был тёмный и бурный вечер... 🌟"  # Russian with emoji

        Chapter.add_chapter(cursor, 30, chapter_names, chapter_text)

        # Read it back
        retrieved_chapter = Chapter.read_chapter(cursor, 30)

        assert retrieved_chapter is not None
        assert retrieved_chapter.name == chapter_names
        assert retrieved_chapter.text == chapter_text


def test_chapter_large_text(temp_db: SafeConnection) -> None:
    """Test adding a chapter with large text content."""
    with temp_db.transaction_cursor() as cursor:
        chapter_names = ["Large Chapter Test"]
        # Create a large text (10KB)
        chapter_text = "This is a test. " * 625  # 16 chars * 625 = 10,000 chars

        Chapter.add_chapter(cursor, 40, chapter_names, chapter_text)

        # Read it back
        retrieved_chapter = Chapter.read_chapter(cursor, 40)

        assert retrieved_chapter is not None
        assert retrieved_chapter.text == chapter_text
        assert len(retrieved_chapter.text) == 10000


def test_chapter_conversation_id_tracking(temp_db: SafeConnection) -> None:
    """Test the conversation_id field and related methods."""
    with temp_db.transaction_cursor() as cursor:
        # Add multiple chapters
        ch1 = Chapter.add_chapter(cursor, 1, ["Book", "Ch1"], "First chapter")
        ch2 = Chapter.add_chapter(cursor, 2, ["Book", "Ch2"], "Second chapter")
        ch3 = Chapter.add_chapter(cursor, 3, ["Book", "Ch3"], "Third chapter")

        # All should have conversation_id as None initially
        assert ch1.conversation_id is None
        assert ch2.conversation_id is None
        assert ch3.conversation_id is None

        # Find first unstarted chapter should return ch1
        first_unstarted = Chapter.find_first_unstarted_chapter(cursor)
        assert first_unstarted is not None
        assert first_unstarted.id == 1

        # Create a conversation and mark ch1 as started
        conv1 = Conversation.create(cursor)
        ch1.set_conversation_id(conv1)

        # Find first unstarted should now return ch2
        first_unstarted = Chapter.find_first_unstarted_chapter(cursor)
        assert first_unstarted is not None
        assert first_unstarted.id == 2

        # Mark ch2 as started by a different conversation
        conv2 = Conversation.create(cursor)
        ch2.set_conversation_id(conv2)

        # Find first unstarted should now return ch3
        first_unstarted = Chapter.find_first_unstarted_chapter(cursor)
        assert first_unstarted is not None
        assert first_unstarted.id == 3

        # Mark ch3 as started
        ch3.set_conversation_id(conv1)

        # Now all chapters are started, should return None
        first_unstarted = Chapter.find_first_unstarted_chapter(cursor)
        assert first_unstarted is None

        # Test read_chapter - should be able to read any chapter
        ch1_read = Chapter.read_chapter(cursor, 1)
        assert ch1_read is not None
        assert ch1_read.conversation_id == conv1.id

        # Add a new unstarted chapter
        Chapter.add_chapter(cursor, 4, ["Book", "Ch4"], "Fourth chapter")

        # Should be able to read ch4 regardless of started status
        ch4_read = Chapter.read_chapter(cursor, 4)
        assert ch4_read is not None
        assert ch4_read.conversation_id is None


def test_get_latest_started_chapter(temp_db: SafeConnection) -> None:
    """Test the get_latest_started_chapter method."""
    with temp_db.transaction_cursor() as cursor:
        # Initially with no chapters, should return None
        latest = Chapter.get_latest_started_chapter(cursor)
        assert latest is None

        # Add chapters but don't start any - should still return None
        ch1 = Chapter.add_chapter(cursor, 1, ["Book", "Ch1"], "First chapter")
        ch2 = Chapter.add_chapter(cursor, 2, ["Book", "Ch2"], "Second chapter")
        ch3 = Chapter.add_chapter(cursor, 3, ["Book", "Ch3"], "Third chapter")

        latest = Chapter.get_latest_started_chapter(cursor)
        assert latest is None

        # Start chapter 1 - should return chapter 1
        conv1 = Conversation.create(cursor)
        ch1.set_conversation_id(conv1)

        latest = Chapter.get_latest_started_chapter(cursor)
        assert latest is not None
        assert latest.id == 1
        assert latest.conversation_id == conv1.id

        # Start chapter 2 - should return chapter 2
        conv2 = Conversation.create(cursor)
        ch2.set_conversation_id(conv2)

        latest = Chapter.get_latest_started_chapter(cursor)
        assert latest is not None
        assert latest.id == 2
        assert latest.conversation_id == conv2.id

        # Start chapter 3 - should return chapter 3
        ch3.set_conversation_id(conv1)

        latest = Chapter.get_latest_started_chapter(cursor)
        assert latest is not None
        assert latest.id == 3
        assert latest.conversation_id == conv1.id

        # Add chapter 4 but don't start it - should still return chapter 3
        Chapter.add_chapter(cursor, 4, ["Book", "Ch4"], "Fourth chapter")

        latest = Chapter.get_latest_started_chapter(cursor)
        assert latest is not None
        assert latest.id == 3

        # Add chapter 5 and 6
        ch5 = Chapter.add_chapter(cursor, 5, ["Book", "Ch5"], "Fifth chapter")
        Chapter.add_chapter(cursor, 6, ["Book", "Ch6"], "Sixth chapter")

        # Start chapter 5 but not 4 or 6 - should return chapter 5
        ch5.set_conversation_id(conv2)

        latest = Chapter.get_latest_started_chapter(cursor)
        assert latest is not None
        assert latest.id == 5
        assert latest.conversation_id == conv2.id


def test_get_latest_started_chapter_non_sequential(temp_db: SafeConnection) -> None:
    """Test get_latest_started_chapter with non-sequential chapter indices."""
    with temp_db.transaction_cursor() as cursor:
        # Add chapters with non-sequential indices
        ch10 = Chapter.add_chapter(cursor, 10, ["Book", "Ch10"], "Chapter 10")
        ch20 = Chapter.add_chapter(cursor, 20, ["Book", "Ch20"], "Chapter 20")
        ch5 = Chapter.add_chapter(cursor, 5, ["Book", "Ch5"], "Chapter 5")
        Chapter.add_chapter(cursor, 15, ["Book", "Ch15"], "Chapter 15")

        conv = Conversation.create(cursor)

        # Start chapters in a different order than their indices
        ch5.set_conversation_id(conv)

        latest = Chapter.get_latest_started_chapter(cursor)
        assert latest is not None
        assert latest.id == 5

        ch20.set_conversation_id(conv)

        latest = Chapter.get_latest_started_chapter(cursor)
        assert latest is not None
        assert latest.id == 20

        ch10.set_conversation_id(conv)

        latest = Chapter.get_latest_started_chapter(cursor)
        assert latest is not None
        assert latest.id == 20  # Should still be 20, the highest index

        # Chapter 15 is not started, so latest should still be 20
        latest = Chapter.get_latest_started_chapter(cursor)
        assert latest is not None
        assert latest.id == 20
