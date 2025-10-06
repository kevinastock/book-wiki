"""Tests for processor chapter finalization behavior."""

from unittest.mock import Mock

import pytest

from bookwiki.db import SafeConnection
from bookwiki.llm import LLMResponse
from bookwiki.models import Chapter, Conversation, WikiPage
from bookwiki.processor import Processor


def create_mock_llm_service() -> Mock:
    """Create a mock LLM service for testing."""
    mock_service = Mock()
    mock_service.get_compression_threshold.return_value = 10000
    mock_service.prompt.return_value = "mock_response_id"
    mock_service.try_fetch.return_value = None
    return mock_service


def test_finalize_chapter_creates_chapter_summary_request_when_missing(
    temp_db: SafeConnection,
) -> None:
    """Test that finalization requests chapter summary when it doesn't exist."""
    with temp_db.transaction_cursor() as cursor:
        # Set up chapter and conversation
        chapter = Chapter.add_chapter(
            cursor,
            chapter_id=1,
            name=["Book", "Chapter 1"],
            text="Chapter content here...",
        )
        conversation = Conversation.create(cursor)
        chapter.set_conversation_id(conversation)

        # Create processor and assistant block
        processor = Processor(temp_db, create_mock_llm_service())
        assistant_block = conversation.add_assistant_text("Chapter complete")

        # Verify no chapter summary exists initially
        summary_page = WikiPage.read_page_at(cursor, "chapter-summary", 1)
        assert summary_page is None

        # Call finalize_chapter
        processor._finalize_chapter(conversation, assistant_block)

        # Verify that a user message was added requesting chapter summary
        blocks = conversation.blocks
        last_block = blocks[-1]
        assert last_block.text_role == "user"
        assert last_block.text_body is not None
        assert "chapter-summary" in last_block.text_body
        assert "summarizes the key events" in last_block.text_body

        # Verify chapter summary page ID is still None
        updated_chapter = Chapter.read_chapter(cursor, 1)
        assert updated_chapter is not None
        assert updated_chapter.chapter_summary_page_id is None


def test_finalize_chapter_links_and_deletes_existing_summary(
    temp_db: SafeConnection,
) -> None:
    """Test that finalization properly handles existing chapter summary."""
    with temp_db.transaction_cursor() as cursor:
        # Set up chapter and conversation
        chapter = Chapter.add_chapter(
            cursor,
            chapter_id=1,
            name=["Book", "Chapter 1"],
            text="Chapter content here...",
        )
        conversation = Conversation.create(cursor)
        chapter.set_conversation_id(conversation)

        # Create assistant block for creating the summary
        create_block = conversation.add_assistant_text("Creating summary")

        # Create chapter summary page
        summary_page = create_block.write_wiki_page(
            chapter_id=1,
            slug="chapter-summary",
            title="Chapter 1 Summary",
            names=["Chapter Summary"],
            summary="Summary of chapter 1",
            body="This chapter introduces the main character.",
        )

        # Create another page that links to the summary
        linking_block = conversation.add_assistant_text("Creating character page")
        linking_block.write_wiki_page(
            chapter_id=1,
            slug="character-page",
            title="Main Character",
            names=["Hero"],
            summary="The protagonist",
            body="See also [Chapter Summary](chapter-summary) for context.",
        )

        # Create processor and final assistant block
        processor = Processor(temp_db, create_mock_llm_service())
        final_assistant_block = conversation.add_assistant_text("Chapter complete")

        # Verify summary exists before finalization
        existing_summary = WikiPage.read_page_at(cursor, "chapter-summary", 1)
        assert existing_summary is not None
        assert existing_summary.title == "Chapter 1 Summary"

        # Call finalize_chapter
        processor._finalize_chapter(conversation, final_assistant_block)

        # Verify chapter was linked to summary page
        updated_chapter = Chapter.read_chapter(cursor, 1)
        assert updated_chapter is not None
        assert updated_chapter.chapter_summary_page_id == summary_page.id

        # Verify summary page was deleted (title set to empty)
        deleted_summary = WikiPage.read_page_at(cursor, "chapter-summary", 1)
        assert deleted_summary is None

        # Verify linking page was updated (links removed)
        updated_linking_page = WikiPage.read_page_at(cursor, "character-page", 1)
        assert updated_linking_page is not None
        assert "Chapter Summary" in updated_linking_page.body  # Display text remains
        assert "(chapter-summary)" not in updated_linking_page.body  # Link removed


def test_finalize_chapter_error_when_no_assistant_block(
    temp_db: SafeConnection,
) -> None:
    """Test that finalization raises error when no assistant block provided."""
    with temp_db.transaction_cursor() as cursor:
        # Set up chapter and conversation
        chapter = Chapter.add_chapter(
            cursor,
            chapter_id=1,
            name=["Book", "Chapter 1"],
            text="Chapter content here...",
        )
        conversation = Conversation.create(cursor)
        chapter.set_conversation_id(conversation)

        # Create a chapter summary page so we get past the early return
        create_block = conversation.add_assistant_text("Creating summary")
        create_block.write_wiki_page(
            chapter_id=1,
            slug="chapter-summary",
            title="Chapter Summary",
            names=["Summary"],
            summary="Summary",
            body="Summary content",
        )

        # Create processor
        processor = Processor(temp_db, create_mock_llm_service())

        # Call finalize_chapter with None block - now should hit the error
        with pytest.raises(ValueError, match="no assistant block available"):
            processor._finalize_chapter(conversation, None)


def test_finalize_chapter_warning_when_no_started_chapter(
    temp_db: SafeConnection,
) -> None:
    """Test that finalization handles case with no started chapter gracefully."""
    with temp_db.transaction_cursor() as cursor:
        # Create conversation but no started chapter
        conversation = Conversation.create(cursor)
        assistant_block = conversation.add_assistant_text("Some text")

        # Create processor
        processor = Processor(temp_db, create_mock_llm_service())

        # Call finalize_chapter - should not raise, just log warning
        processor._finalize_chapter(conversation, assistant_block)

        # No exception should be raised, method should return gracefully


def test_finalize_chapter_sets_chapter_summary_page_id(temp_db: SafeConnection) -> None:
    """Test that chapter.set_chapter_summary_page works correctly."""
    with temp_db.transaction_cursor() as cursor:
        # Set up chapter and conversation
        chapter = Chapter.add_chapter(
            cursor,
            chapter_id=1,
            name=["Book", "Chapter 1"],
            text="Chapter content here...",
        )
        conversation = Conversation.create(cursor)
        chapter.set_conversation_id(conversation)

        # Create a wiki page
        create_block = conversation.add_assistant_text("Creating page")
        wiki_page = create_block.write_wiki_page(
            chapter_id=1,
            slug="test-page",
            title="Test Page",
            names=["Test"],
            summary="A test page",
            body="Test content",
        )

        # Initially no summary page ID
        assert chapter.chapter_summary_page_id is None

        # Set the summary page
        chapter.set_chapter_summary_page(wiki_page)

        # Verify it was set in the database
        updated_chapter = Chapter.read_chapter(cursor, 1)
        assert updated_chapter is not None
        assert updated_chapter.chapter_summary_page_id == wiki_page.id

        # Verify the cached property works
        summary_page = updated_chapter.chapter_summary_page
        assert summary_page is not None
        assert summary_page.id == wiki_page.id
        assert summary_page.slug == "test-page"


def test_processor_integration_with_finalize_chapter(temp_db: SafeConnection) -> None:
    """Test integration of _finalize_chapter with _retrieve_and_handle_conversation."""

    def create_integration_mock_llm_service(texts: list[str]) -> Mock:
        """Create mock LLM service that returns a specific response."""
        mock_service = Mock()
        mock_service.get_compression_threshold.return_value = 10000
        mock_service.prompt.return_value = "mock_response_id"

        mock_response = LLMResponse(
            tools=[],
            texts=texts,
            updated_prev="updated_prev_id",
            compressing=False,
            input_tokens=100,
            output_tokens=50,
        )
        mock_service.try_fetch.return_value = mock_response
        return mock_service

    with temp_db.transaction_cursor() as cursor:
        # Set up chapter and conversation
        chapter = Chapter.add_chapter(
            cursor,
            chapter_id=1,
            name=["Book", "Chapter 1"],
            text="Chapter content here...",
        )
        conversation = Conversation.create(cursor)
        chapter.set_conversation_id(conversation)

        # Add user message and set conversation as waiting
        conversation.add_user_text("Please analyze this chapter")
        conversation.set_waiting_on_id("mock_response_id")

        # Create mock response with no tools (triggers finalization)
        processor = Processor(
            temp_db,
            create_integration_mock_llm_service(
                ["Analysis complete. The chapter introduces key themes."]
            ),
        )

        # Get fresh conversation object with correct waiting_on_id
        fresh_conversation = Conversation.get_by_id(cursor, conversation.id)
        assert fresh_conversation is not None
        assert fresh_conversation.waiting_on_id == "mock_response_id"

        # Call _retrieve_and_handle_conversation (which should call _finalize_chapter)
        processor._retrieve_and_handle_conversation(fresh_conversation)

        # Get fresh conversation to see updated blocks
        updated_conversation = Conversation.get_by_id(cursor, conversation.id)
        assert updated_conversation is not None

        # Verify that finalization was triggered (user message added for summary)
        blocks = updated_conversation.blocks

        # Should have: original user message, assistant response,
        # finalization user message
        assert len(blocks) >= 3

        # Find the last user message (should be the finalization request)
        user_blocks = [b for b in blocks if b.text_role == "user"]
        assert len(user_blocks) >= 2  # Original + finalization request

        last_user_block = user_blocks[-1]
        assert last_user_block.text_body is not None
        assert "chapter-summary" in last_user_block.text_body
        assert "summarizes the key events" in last_user_block.text_body

        # Verify conversation state was properly updated
        assert updated_conversation.waiting_on_id is None  # Cleared after processing
        assert updated_conversation.total_input_tokens == 100
        assert updated_conversation.total_output_tokens == 50


def test_finalize_chapter_with_existing_summary_and_complex_links(
    temp_db: SafeConnection,
) -> None:
    """Test finalization with complex wiki link scenarios."""
    with temp_db.transaction_cursor() as cursor:
        # Set up chapter and conversation
        chapter = Chapter.add_chapter(
            cursor,
            chapter_id=1,
            name=["Book", "Chapter 1"],
            text="Chapter content here...",
        )
        conversation = Conversation.create(cursor)
        chapter.set_conversation_id(conversation)

        # Create assistant block and summary
        create_block = conversation.add_assistant_text("Creating summary")
        summary_page = create_block.write_wiki_page(
            chapter_id=1,
            slug="chapter-summary",
            title="Chapter 1 Summary",
            names=["Summary"],
            summary="Chapter summary",
            body="This chapter covers important events.",
        )

        # Create multiple pages with different link formats
        link_block1 = conversation.add_assistant_text("Creating page 1")
        link_block1.write_wiki_page(
            chapter_id=1,
            slug="page-1",
            title="Page 1",
            names=["Page One"],
            summary="First page",
            body=(
                "See [overview](chapter-summary) for context. "
                "Also check [details](chapter-summary)."
            ),
        )

        link_block2 = conversation.add_assistant_text("Creating page 2")
        link_block2.write_wiki_page(
            chapter_id=1,
            slug="page-2",
            title="Page 2",
            names=["Page Two"],
            summary="Second page",
            body="No links to summary here, but mentions other topics.",
        )

        # Create processor and finalize
        processor = Processor(temp_db, create_mock_llm_service())
        final_block = conversation.add_assistant_text("Chapter complete")
        processor._finalize_chapter(conversation, final_block)

        # Verify chapter linked to summary
        updated_chapter = Chapter.read_chapter(cursor, 1)
        assert updated_chapter is not None
        assert updated_chapter.chapter_summary_page_id == summary_page.id

        # Verify summary was deleted
        assert WikiPage.read_page_at(cursor, "chapter-summary", 1) is None

        # Verify page 1 had links removed but display text preserved
        updated_page1 = WikiPage.read_page_at(cursor, "page-1", 1)
        assert updated_page1 is not None
        assert "overview" in updated_page1.body  # Display text preserved
        assert "details" in updated_page1.body  # Display text preserved
        assert "(chapter-summary)" not in updated_page1.body  # Links removed

        # Verify page 2 was unchanged (no links to remove)
        updated_page2 = WikiPage.read_page_at(cursor, "page-2", 1)
        assert updated_page2 is not None
        assert "No links to summary" in updated_page2.body
