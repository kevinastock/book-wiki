"""Tests for bookwiki chapter tools."""

import pytest

from bookwiki.db import SafeConnection
from bookwiki.models import Block, Chapter, Conversation
from bookwiki.tools.chapter import ReadChapter


def test_read_chapter_success(temp_db: SafeConnection) -> None:
    """Test successfully reading an existing chapter."""
    with temp_db.transaction_cursor() as cursor:
        # Set up test data
        chapter = Chapter.add_chapter(
            cursor,
            chapter_id=1,
            name=["Book One", "Chapter 1", "The Beginning"],
            text="Once upon a time, in a land far away...",
        )

        conversation = Conversation.create(cursor)
        # Mark the chapter as started by this conversation
        chapter.set_conversation_id(conversation)
        block = Block.create_tool_use(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            name="ReadChapter",
            use_id="read_1",
            params='{"chapter_offset": 0}',
        )

        # Create and apply the tool
        tool = ReadChapter(tool_id="read_1", tool_name="ReadChapter", chapter_offset=0)

        tool.apply(block)

        # Check the response
        updated_block = Block.get_by_id(cursor, block.id)
        assert updated_block is not None
        expected_response = (
            "**Book One > Chapter 1 > The Beginning**\n\n"
            "Once upon a time, in a land far away..."
        )
        assert updated_block.tool_response == expected_response
        assert updated_block.errored is False


def test_read_chapter_nonexistent(temp_db: SafeConnection) -> None:
    """Test reading a chapter that doesn't exist."""
    with temp_db.transaction_cursor() as cursor:
        # Add and start chapter 1
        ch1 = Chapter.add_chapter(
            cursor,
            chapter_id=1,
            name=["Book One", "Chapter 1"],
            text="First chapter content.",
        )
        conversation = Conversation.create(cursor)
        ch1.set_conversation_id(conversation)

        block = Block.create_tool_use(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            name="ReadChapter",
            use_id="read_missing",
            params='{"chapter_offset": -999}',
        )

        # Create and apply the tool
        tool = ReadChapter(
            tool_id="read_missing", tool_name="ReadChapter", chapter_offset=-999
        )

        tool.apply(block)

        # Should get an error response
        updated_block = Block.get_by_id(cursor, block.id)
        assert updated_block is not None
        assert updated_block.tool_response is not None
        assert "does not exist" in updated_block.tool_response
        assert updated_block.errored is True


def test_read_chapter_multiple_chapters(temp_db: SafeConnection) -> None:
    """Test reading different chapters using offsets."""
    with temp_db.transaction_cursor() as cursor:
        # Add multiple chapters
        ch1 = Chapter.add_chapter(
            cursor,
            chapter_id=1,
            name=["Book One", "Chapter 1"],
            text="First chapter content.",
        )
        ch2 = Chapter.add_chapter(
            cursor,
            chapter_id=2,
            name=["Book One", "Chapter 2"],
            text="Second chapter content.",
        )
        ch5 = Chapter.add_chapter(
            cursor,
            chapter_id=5,
            name=["Book One", "Chapter 5"],
            text="Fifth chapter content.",
        )

        conversation = Conversation.create(cursor)
        # Mark all chapters as started
        ch1.set_conversation_id(conversation)
        ch2.set_conversation_id(conversation)
        ch5.set_conversation_id(conversation)

        # Read latest chapter (chapter 5 with offset=0)
        block_latest = Block.create_tool_use(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            name="ReadChapter",
            use_id="read_latest",
            params='{"chapter_offset": 0}',
        )

        tool_latest = ReadChapter(
            tool_id="read_latest", tool_name="ReadChapter", chapter_offset=0
        )
        tool_latest.apply(block_latest)

        # Read chapter 2 (with offset=-3 from chapter 5)
        block2 = Block.create_tool_use(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            name="ReadChapter",
            use_id="read_2",
            params='{"chapter_offset": -3}',
        )

        tool2 = ReadChapter(
            tool_id="read_2", tool_name="ReadChapter", chapter_offset=-3
        )
        tool2.apply(block2)

        # Check responses
        updated_block_latest = Block.get_by_id(cursor, block_latest.id)
        assert updated_block_latest is not None
        assert updated_block_latest.tool_response is not None
        assert "**Book One > Chapter 5**" in updated_block_latest.tool_response
        assert "Fifth chapter content." in updated_block_latest.tool_response

        updated_block2 = Block.get_by_id(cursor, block2.id)
        assert updated_block2 is not None
        assert updated_block2.tool_response is not None
        assert "**Book One > Chapter 2**" in updated_block2.tool_response
        assert "Second chapter content." in updated_block2.tool_response


def test_cannot_add_chapter_with_empty_name_list(temp_db: SafeConnection) -> None:
    """Test that adding a chapter with empty name list is prevented."""
    with (
        temp_db.transaction_cursor() as cursor,
        pytest.raises(ValueError, match="Chapter name list cannot be empty"),
    ):
        Chapter.add_chapter(
            cursor, chapter_id=10, name=[], text="Content with no chapter name."
        )


def test_read_chapter_unicode_content(temp_db: SafeConnection) -> None:
    """Test reading a chapter with unicode content."""
    with temp_db.transaction_cursor() as cursor:
        # Add chapter with unicode
        ch = Chapter.add_chapter(
            cursor,
            chapter_id=20,
            name=["Книга", "Глава 1", "Начало"],
            text="Жил-был дракон 🐉 в далёкой стране...",
        )

        conversation = Conversation.create(cursor)
        # Mark chapter as started
        ch.set_conversation_id(conversation)
        block = Block.create_tool_use(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            name="ReadChapter",
            use_id="read_unicode",
            params='{"chapter_offset": 0}',
        )

        tool = ReadChapter(
            tool_id="read_unicode", tool_name="ReadChapter", chapter_offset=0
        )

        tool.apply(block)

        # Check the response preserves unicode
        updated_block = Block.get_by_id(cursor, block.id)
        assert updated_block is not None
        assert updated_block.tool_response is not None
        assert "**Книга > Глава 1 > Начало**" in updated_block.tool_response
        assert "Жил-был дракон 🐉" in updated_block.tool_response


def test_read_chapter_require_started(temp_db: SafeConnection) -> None:
    """Test that ReadChapter only allows reading chapters that have been started."""
    with temp_db.transaction_cursor() as cursor:
        # Add chapters 29 and 30
        ch29 = Chapter.add_chapter(
            cursor,
            chapter_id=29,
            name=["Book", "Chapter 29"],
            text="Chapter 29 content.",
        )
        ch30 = Chapter.add_chapter(
            cursor,
            chapter_id=30,
            name=["Book", "Chapter 30"],
            text="Chapter 30 content.",
        )

        conversation = Conversation.create(cursor)
        # Start only chapter 29
        ch29.set_conversation_id(conversation)

        # Try to read chapter 30 (not started) with offset=1
        block = Block.create_tool_use(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            name="ReadChapter",
            use_id="read_unstarted",
            params='{"chapter_offset": 1}',
        )

        tool = ReadChapter(
            tool_id="read_unstarted", tool_name="ReadChapter", chapter_offset=1
        )
        tool.apply(block)

        # Should get an error about reading future chapters
        updated_block = Block.get_by_id(cursor, block.id)
        assert updated_block is not None
        assert updated_block.tool_response is not None
        assert "Cannot read future chapters" in updated_block.tool_response
        assert updated_block.errored is True

        # Now mark chapter 30 as started
        ch30.set_conversation_id(conversation)

        # Try to read again with offset=0 (should work now)
        block2 = Block.create_tool_use(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            name="ReadChapter",
            use_id="read_started",
            params='{"chapter_offset": 0}',
        )

        tool2 = ReadChapter(
            tool_id="read_started", tool_name="ReadChapter", chapter_offset=0
        )
        tool2.apply(block2)

        # Should succeed now
        updated_block2 = Block.get_by_id(cursor, block2.id)
        assert updated_block2 is not None
        assert updated_block2.tool_response is not None
        assert "**Book > Chapter 30**" in updated_block2.tool_response
        assert "Chapter 30 content." in updated_block2.tool_response
        assert updated_block2.errored is False


def test_read_chapter_latest_started(temp_db: SafeConnection) -> None:
    """Test reading latest started chapter when chapter_offset is 0 or not provided."""
    with temp_db.transaction_cursor() as cursor:
        # Add multiple chapters
        ch1 = Chapter.add_chapter(
            cursor,
            chapter_id=1,
            name=["Book One", "Chapter 1"],
            text="First chapter content.",
        )
        ch2 = Chapter.add_chapter(
            cursor,
            chapter_id=2,
            name=["Book One", "Chapter 2"],
            text="Second chapter content.",
        )
        Chapter.add_chapter(
            cursor,
            chapter_id=3,
            name=["Book One", "Chapter 3"],
            text="Third chapter content.",
        )

        conversation = Conversation.create(cursor)
        # Start only chapters 1 and 2, not 3
        ch1.set_conversation_id(conversation)
        ch2.set_conversation_id(conversation)  # Chapter 2 is the latest started

        # Read without specifying chapter_offset (default to 0 = latest = ch2)
        block = Block.create_tool_use(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            name="ReadChapter",
            use_id="read_latest",
            params="{}",  # No chapter_offset provided
        )

        tool = ReadChapter(
            tool_id="read_latest", tool_name="ReadChapter"
        )  # chapter_offset defaults to 0
        tool._apply(block)

        # Should get chapter 2 content
        updated_block = Block.get_by_id(cursor, block.id)
        assert updated_block is not None
        assert updated_block.tool_response is not None
        assert "**Book One > Chapter 2**" in updated_block.tool_response
        assert "Second chapter content." in updated_block.tool_response


def test_read_chapter_no_chapters_started_with_none_id(temp_db: SafeConnection) -> None:
    """Test reading when no chapters are started."""
    with temp_db.transaction_cursor() as cursor:
        # Add chapters but don't start any
        Chapter.add_chapter(
            cursor,
            chapter_id=1,
            name=["Book One", "Chapter 1"],
            text="First chapter content.",
        )
        Chapter.add_chapter(
            cursor,
            chapter_id=2,
            name=["Book One", "Chapter 2"],
            text="Second chapter content.",
        )

        conversation = Conversation.create(cursor)

        # Try to read with default offset when no chapters are started
        block = Block.create_tool_use(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            name="ReadChapter",
            use_id="read_none_started",
            params="{}",  # No chapter_offset provided
        )

        tool = ReadChapter(
            tool_id="read_none_started", tool_name="ReadChapter"
        )  # chapter_offset defaults to 0

        tool.apply(block)

        # Should get an error
        updated_block = Block.get_by_id(cursor, block.id)
        assert updated_block is not None
        assert updated_block.tool_response is not None
        assert "No chapters have been started yet" in updated_block.tool_response
        assert updated_block.errored is True


def test_read_chapter_beyond_latest_started(temp_db: SafeConnection) -> None:
    """Test that positive offsets (reading future chapters) are not allowed."""
    with temp_db.transaction_cursor() as cursor:
        # Add multiple chapters
        ch1 = Chapter.add_chapter(
            cursor,
            chapter_id=1,
            name=["Book One", "Chapter 1"],
            text="First chapter content.",
        )
        Chapter.add_chapter(
            cursor,
            chapter_id=2,
            name=["Book One", "Chapter 2"],
            text="Second chapter content.",
        )
        Chapter.add_chapter(
            cursor,
            chapter_id=3,
            name=["Book One", "Chapter 3"],
            text="Third chapter content.",
        )

        conversation = Conversation.create(cursor)
        # Start only chapter 1
        ch1.set_conversation_id(conversation)

        # Try to read with positive offset (would be chapter 2 or 3)
        block = Block.create_tool_use(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            name="ReadChapter",
            use_id="read_future",
            params='{"chapter_offset": 2}',
        )

        tool = ReadChapter(
            tool_id="read_future", tool_name="ReadChapter", chapter_offset=2
        )

        tool.apply(block)

        # Should get an error about reading future chapters
        updated_block = Block.get_by_id(cursor, block.id)
        assert updated_block is not None
        assert updated_block.tool_response is not None
        assert "Cannot read future chapters" in updated_block.tool_response
        assert updated_block.errored is True


def test_read_chapter_validation() -> None:
    """Test ReadChapter parameter validation."""
    # Valid initialization with default offset
    tool_default = ReadChapter(tool_id="test", tool_name="ReadChapter")
    assert tool_default.chapter_offset is None

    # Test with negative offset (allowed for reading previous chapters)
    tool_negative = ReadChapter(
        tool_id="test_neg", tool_name="ReadChapter", chapter_offset=-1
    )
    assert tool_negative.chapter_offset == -1

    # Test with zero (current chapter)
    tool_zero = ReadChapter(
        tool_id="test_zero", tool_name="ReadChapter", chapter_offset=0
    )
    assert tool_zero.chapter_offset == 0

    # Test with positive offset (will be rejected during apply)
    tool_positive = ReadChapter(
        tool_id="test_pos", tool_name="ReadChapter", chapter_offset=1
    )
    assert tool_positive.chapter_offset == 1


def test_read_chapter_none_offset_equals_zero(temp_db: SafeConnection) -> None:
    """Test that None offset is treated the same as 0 offset."""
    with temp_db.transaction_cursor() as cursor:
        # Add and start a chapter
        ch1 = Chapter.add_chapter(
            cursor,
            chapter_id=1,
            name=["Book One", "Chapter 1"],
            text="First chapter content.",
        )
        conversation = Conversation.create(cursor)
        ch1.set_conversation_id(conversation)

        # Read with None offset (should read latest chapter)
        block_none = Block.create_tool_use(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            name="ReadChapter",
            use_id="read_none",
            params="{}",  # No chapter_offset provided
        )

        tool_none = ReadChapter(
            tool_id="read_none", tool_name="ReadChapter"
        )  # chapter_offset=None
        tool_none.apply(block_none)

        # Read with 0 offset (should read latest chapter)
        block_zero = Block.create_tool_use(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            name="ReadChapter",
            use_id="read_zero",
            params='{"chapter_offset": 0}',
        )

        tool_zero = ReadChapter(
            tool_id="read_zero", tool_name="ReadChapter", chapter_offset=0
        )
        tool_zero.apply(block_zero)

        # Both should produce identical results
        updated_block_none = Block.get_by_id(cursor, block_none.id)
        updated_block_zero = Block.get_by_id(cursor, block_zero.id)

        assert updated_block_none is not None
        assert updated_block_zero is not None
        assert updated_block_none.tool_response is not None
        assert updated_block_zero.tool_response is not None
        assert updated_block_none.tool_response == updated_block_zero.tool_response
        assert not updated_block_none.errored
        assert not updated_block_zero.errored


def test_read_chapter_with_negative_offset(temp_db: SafeConnection) -> None:
    """Test reading previous chapters using negative offset."""
    with temp_db.transaction_cursor() as cursor:
        # Add multiple chapters
        ch1 = Chapter.add_chapter(
            cursor,
            chapter_id=1,
            name=["Book One", "Chapter 1"],
            text="First chapter content.",
        )
        ch2 = Chapter.add_chapter(
            cursor,
            chapter_id=2,
            name=["Book One", "Chapter 2"],
            text="Second chapter content.",
        )
        ch3 = Chapter.add_chapter(
            cursor,
            chapter_id=3,
            name=["Book One", "Chapter 3"],
            text="Third chapter content.",
        )

        conversation = Conversation.create(cursor)
        # Start all chapters
        ch1.set_conversation_id(conversation)
        ch2.set_conversation_id(conversation)
        ch3.set_conversation_id(conversation)  # Chapter 3 is latest

        # Read chapter 1 with offset=-2 from chapter 3
        block = Block.create_tool_use(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            name="ReadChapter",
            use_id="read_ch1",
            params='{"chapter_offset": -2}',
        )

        tool = ReadChapter(
            tool_id="read_ch1", tool_name="ReadChapter", chapter_offset=-2
        )
        tool.apply(block)

        # Should get chapter 1 content
        updated_block = Block.get_by_id(cursor, block.id)
        assert updated_block is not None
        assert updated_block.tool_response is not None
        assert "**Book One > Chapter 1**" in updated_block.tool_response
        assert "First chapter content." in updated_block.tool_response
        assert updated_block.errored is False
