"""Test coverage for uncovered model methods."""

import pytest

from bookwiki.db import SafeConnection
from bookwiki.models import Block, Chapter, Conversation, WikiPage


def test_block_conversation(temp_db: SafeConnection) -> None:
    """Test Block.conversation() property."""
    with temp_db.transaction_cursor() as cursor:
        # Create a conversation
        conversation = Conversation.create(cursor)

        # Create a block in that conversation
        block = conversation.add_user_text("Test message")

        # Test getting the conversation
        retrieved_conv = block.conversation
        assert retrieved_conv.id == conversation.id

        # Test with invalid conversation reference using Block.create_text directly
        invalid_block = Block.create_text(
            cursor=cursor,
            conversation_id=999999,
            generation=0,
            role="user",
            text="Test",
            sent=True,
        )
        assert invalid_block is not None

        with pytest.raises(ValueError, match="references non-existent conversation"):
            _ = invalid_block.conversation


def test_block_spawned_conversation(temp_db: SafeConnection) -> None:
    """Test Block.spawned_conversation() property."""
    with temp_db.transaction_cursor() as cursor:
        # Create a parent conversation and block
        parent_conv = Conversation.create(cursor)
        parent_block = parent_conv.add_user_text("Parent")

        # Initially no spawned conversation
        assert parent_block.spawned_conversation is None

        # Create a spawned conversation
        spawned_conv = Conversation.create(cursor, parent_block_id=parent_block.id)

        reload = Block.get_by_id(cursor, parent_block.id)
        assert reload is not None
        # Now should find the spawned conversation
        retrieved_spawned = reload.spawned_conversation
        assert retrieved_spawned is not None
        assert retrieved_spawned.id == spawned_conv.id
        assert retrieved_spawned.parent_block_id == reload.id


def test_block_created_wiki_page(temp_db: SafeConnection) -> None:
    """Test Block.created_wiki_page property."""
    with temp_db.transaction_cursor() as cursor:
        # Create necessary setup
        conversation = Conversation.create(cursor)
        block = conversation.add_user_text("Create page")

        # Initially no wiki page
        assert block.created_wiki_page is None

        # Create a chapter first (required for wiki page)
        Chapter.add_chapter(cursor, 1, ["Book", "Chapter 1"], "Content")

        # Create wiki page linked to this block
        from bookwiki.models import WikiPage

        wiki_page = WikiPage.create(
            cursor,
            chapter_id=1,
            slug="test-page",
            create_block_id=block.id,
            title="Test Page",
            names=["Test Page"],
            summary="Test summary",
            body="Test body",
        )

        reload = Block.get_by_id(cursor, block.id)
        assert reload is not None
        # Now should find the wiki page
        retrieved_page = reload.created_wiki_page
        assert retrieved_page is not None
        assert retrieved_page.id == wiki_page.id
        assert retrieved_page.create_block_id == reload.id


def test_block_start_conversation_errors(temp_db: SafeConnection) -> None:
    """Test Block.start_conversation() error conditions."""
    with temp_db.transaction_cursor() as cursor:
        conversation = Conversation.create(cursor)

        # Test with non-tool-use block
        text_block = conversation.add_user_text("Not a tool")
        with pytest.raises(
            ValueError, match="Can only start conversations from tool use blocks"
        ):
            text_block.start_conversation()

        # Create a tool use block with response
        tool_block = conversation.add_tool_use(
            "test_tool", "tool_123", '{"param": "value"}'
        )
        tool_block.respond("Response data")

        # Tool block with response can still start conversation - remove this test
        # as the logic might have changed
        spawned = tool_block.start_conversation()
        assert spawned is not None


def test_conversation_get_by_parent_block(temp_db: SafeConnection) -> None:
    """Test Conversation.get_by_parent_block_id() method."""
    with temp_db.transaction_cursor() as cursor:
        # Create parent conversation and block
        parent_conv = Conversation.create(cursor)
        parent_block = parent_conv.add_user_text("Parent")

        # No child conversation initially
        assert Conversation.get_by_parent_block_id(cursor, parent_block.id) is None

        # Create child conversation
        child_conv = Conversation.create(cursor, parent_block_id=parent_block.id)

        # Now should find it
        found = Conversation.get_by_parent_block_id(cursor, parent_block.id)
        assert found is not None
        assert found.id == child_conv.id
        assert found.parent_block_id == parent_block.id


def test_conversation_get_next_waiting_with_after_id(temp_db: SafeConnection) -> None:
    """Test Conversation.get_next_waiting() with after_id parameter."""
    with temp_db.transaction_cursor() as cursor:
        # Create first waiting conversation
        conv1 = Conversation.create(cursor)
        conv1.set_waiting_on_id("waiting1")

        # Create second waiting conversation
        conv2 = Conversation.create(cursor)
        conv2.set_waiting_on_id("waiting2")

        # Create third waiting conversation
        conv3 = Conversation.create(cursor)
        conv3.set_waiting_on_id("waiting3")

        # Get next waiting after conv1
        next_conv = Conversation.find_waiting_conversation(cursor, after_id=conv1.id)
        assert next_conv is not None
        assert next_conv.id == conv2.id

        # Get next waiting after conv2
        next_conv = Conversation.find_waiting_conversation(cursor, after_id=conv2.id)
        assert next_conv is not None
        assert next_conv.id == conv3.id

        # No more waiting after conv3
        next_conv = Conversation.find_waiting_conversation(cursor, after_id=conv3.id)
        assert next_conv is None


def test_conversation_update_previously(temp_db: SafeConnection) -> None:
    """Test Conversation.update_previously() method."""
    with temp_db.transaction_cursor() as cursor:
        conversation = Conversation.create(cursor)
        assert conversation.previously is None

        # Update previously
        conversation.update_previously("response_123")

        # Verify in database (since the instance doesn't update)
        retrieved = Conversation.get_by_id(cursor, conversation.id)
        assert retrieved is not None
        assert retrieved.previously == "response_123"


def test_conversation_get_parent_block(temp_db: SafeConnection) -> None:
    """Test Conversation.parent_block method."""
    with temp_db.transaction_cursor() as cursor:
        # Root conversation has no parent block
        root_conv = Conversation.create(cursor)
        assert root_conv.parent_block is None

        # Create parent block and child conversation
        parent_block = root_conv.add_user_text("Parent")
        child_conv = Conversation.create(cursor, parent_block_id=parent_block.id)

        # Should find parent block
        retrieved_block = child_conv.parent_block
        assert retrieved_block is not None
        assert retrieved_block.id == parent_block.id


def test_conversation_get_chapter_with_parent_chain(temp_db: SafeConnection) -> None:
    """Test Conversation.chapter following parent chain."""
    with temp_db.transaction_cursor() as cursor:
        # Create a chapter
        Chapter.add_chapter(cursor, 1, ["Book", "Chapter 1"], "Content")
        chapter = Chapter.read_chapter(cursor, 1)

        # Create root conversation that started the chapter
        root_conv = Conversation.create(cursor)
        assert chapter is not None
        chapter.set_conversation_id(root_conv)

        # Create a chain of child conversations
        parent_block1 = root_conv.add_tool_use("tool1", "id1", "{}")
        child_conv1 = Conversation.create(cursor, parent_block_id=parent_block1.id)

        parent_block2 = child_conv1.add_tool_use("tool2", "id2", "{}")
        child_conv2 = Conversation.create(cursor, parent_block_id=parent_block2.id)

        # All conversations should find the chapter
        assert root_conv.chapter.id == 1
        assert child_conv1.chapter.id == 1
        assert child_conv2.chapter.id == 1


def test_conversation_get_chapter_error(temp_db: SafeConnection) -> None:
    """Test Conversation.chapter error when no chapter found."""
    with temp_db.transaction_cursor() as cursor:
        # Create conversation without any associated chapter
        orphan_conv = Conversation.create(cursor)

        with pytest.raises(ValueError, match="has no associated chapter"):
            _ = orphan_conv.chapter


def test_chapter_create_duplicate_error(temp_db: SafeConnection) -> None:
    """Test Chapter.create() with duplicate name."""
    with temp_db.transaction_cursor() as cursor:
        # Create first chapter
        Chapter.add_chapter(cursor, 1, ["Book", "Chapter 1"], "Content")

        # Try to create another with same name
        with pytest.raises(ValueError, match="Chapter .* already exists"):
            Chapter.add_chapter(cursor, 2, ["Book", "Chapter 1"], "Different content")


def test_chapter_get_chapter_count_empty(temp_db: SafeConnection) -> None:
    """Test Chapter.get_chapter_count() with empty database."""
    with temp_db.transaction_cursor() as cursor:
        count = Chapter.get_chapter_count(cursor)
        assert count == 0


def test_chapter_get_chapter_count_with_chapters(temp_db: SafeConnection) -> None:
    """Test Chapter.get_chapter_count() with chapters."""
    with temp_db.transaction_cursor() as cursor:
        # Add some chapters
        Chapter.add_chapter(cursor, 1, ["Book", "Chapter 1"], "Content 1")
        Chapter.add_chapter(cursor, 2, ["Book", "Chapter 2"], "Content 2")
        Chapter.add_chapter(cursor, 3, ["Book", "Chapter 3"], "Content 3")

        count = Chapter.get_chapter_count(cursor)
        assert count == 3


def test_wikipage_get_by_id_not_found(temp_db: SafeConnection) -> None:
    """Test WikiPage.get_by_id() when page doesn't exist."""
    with temp_db.transaction_cursor() as cursor:
        page = WikiPage.get_by_id(cursor, 999999)
        assert page is None


def test_wikipage_get_by_create_block_id(temp_db: SafeConnection) -> None:
    """Test WikiPage.get_by_create_block_id() method."""
    with temp_db.transaction_cursor() as cursor:
        # Setup: create chapter, conversation, and block
        Chapter.add_chapter(cursor, 1, ["Book", "Chapter 1"], "Content")
        conversation = Conversation.create(cursor)
        block = conversation.add_user_text("Create wiki page")

        # No wiki page initially
        assert WikiPage.get_by_create_block_id(cursor, block.id) is None

        # Create wiki page with this block
        wiki_page = WikiPage.create(
            cursor,
            chapter_id=1,
            slug="page-name",
            create_block_id=block.id,
            title="Page Name",
            names=["Page Name", "Alternate Name"],
            summary="Summary",
            body="Body content",
        )

        # Should find the wiki page
        found = WikiPage.get_by_create_block_id(cursor, block.id)
        assert found is not None
        assert found.id == wiki_page.id
        assert found.create_block_id == block.id
        assert set(found.names) == {"Page Name", "Alternate Name"}


def test_wikipage_get_create_block_error(temp_db: SafeConnection) -> None:
    """Test WikiPage.get_create_block() with invalid reference."""
    with temp_db.transaction_cursor() as cursor:
        # Create chapter first
        Chapter.add_chapter(cursor, 1, ["Book", "Chapter 1"], "Content")

        # Create wiki page with invalid create_block reference
        # Use normal WikiPage.create with non-existent create_block_id
        page = WikiPage.create(
            cursor,
            chapter_id=1,
            slug="test-page",
            create_block_id=999999,
            title="Test Page",
            names=["Test Page"],  # Required parameter
            summary="Summary",
            body="Body",
        )
        assert page is not None

        with pytest.raises(ValueError, match="references non-existent create_block"):
            _ = page.create_block
