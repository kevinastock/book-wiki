"""Tests for the linting functionality integrated into WriteWikiPage tool."""

import pytest

from bookwiki.db import SafeConnection
from bookwiki.models import Block, Chapter, Conversation
from bookwiki.tools.base import LLMSolvableError
from bookwiki.tools.wiki import WriteWikiPage


def test_write_page_no_lint_issues(temp_db: SafeConnection) -> None:
    """Test writing a page with no broken links."""
    with temp_db.transaction_cursor() as cursor:
        # Create a chapter
        chapter = Chapter.add_chapter(cursor, 1, ["Chapter 1"], "Some content")
        conversation = Conversation.create(cursor)
        chapter.set_conversation_id(conversation)

        # Create block and write page with no links
        block = Block.create_tool_use(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            name="WriteWikiPage",
            use_id="write_1",
            params='{"slug": "gandalf", "title": "Gandalf", "names": ["Gandalf"], '
            '"summary": "A wizard", "body": "A powerful wizard.", "create": true}',
        )

        tool = WriteWikiPage(
            tool_id="write_1",
            tool_name="WriteWikiPage",
            slug="gandalf",
            title="Gandalf",
            names=["Gandalf"],
            summary="A wizard",
            body="A powerful wizard.",
            create=True,
        )
        tool._apply(block)

        # Check response - should just say wrote wiki page
        updated_block = Block.get_by_id(cursor, block.id)
        assert updated_block is not None
        assert updated_block.tool_response == "Wrote wiki page"


def test_write_page_with_valid_links(temp_db: SafeConnection) -> None:
    """Test writing a page with valid wiki links."""
    with temp_db.transaction_cursor() as cursor:
        # Create a chapter
        chapter = Chapter.add_chapter(cursor, 1, ["Chapter 1"], "Some content")
        conversation = Conversation.create(cursor)
        chapter.set_conversation_id(conversation)

        # First create gandalf page
        block1 = Block.create_text(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            role="assistant",
            text="Creating gandalf",
        )
        block1.write_wiki_page(
            chapter_id=chapter.id,
            slug="gandalf",
            title="Gandalf",
            names=["Gandalf"],
            summary="A wizard",
            body="A wizard.",
        )

        # Now create frodo page with link to gandalf
        block = Block.create_tool_use(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            name="WriteWikiPage",
            use_id="write_1",
            params='{"slug": "frodo", "title": "Frodo", "names": ["Frodo"], '
            '"summary": "A hobbit", '
            '"body": "A hobbit who knows [gandalf](gandalf).", "create": true}',
        )

        tool = WriteWikiPage(
            tool_id="write_1",
            tool_name="WriteWikiPage",
            slug="frodo",
            title="Frodo",
            names=["Frodo"],
            summary="A hobbit",
            body="A hobbit who knows [gandalf](gandalf).",
            create=True,
        )
        tool._apply(block)

        # Check response - no errors since gandalf exists
        updated_block = Block.get_by_id(cursor, block.id)
        assert updated_block is not None
        assert updated_block.tool_response == "Wrote wiki page"


def test_write_page_with_broken_links(temp_db: SafeConnection) -> None:
    """Test writing a page with broken wiki links raises error."""
    with temp_db.transaction_cursor() as cursor:
        # Create a chapter
        chapter = Chapter.add_chapter(cursor, 1, ["Chapter 1"], "Some content")
        conversation = Conversation.create(cursor)
        chapter.set_conversation_id(conversation)

        # Create frodo page with links to non-existent pages
        block = Block.create_tool_use(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            name="WriteWikiPage",
            use_id="write_1",
            params='{"slug": "frodo", "title": "Frodo", "names": ["Frodo"], '
            '"summary": "A hobbit", '
            '"body": "A hobbit who knows [gandalf](gandalf) and [aragorn](aragorn).", '
            '"create": true}',
        )

        tool = WriteWikiPage(
            tool_id="write_1",
            tool_name="WriteWikiPage",
            slug="frodo",
            title="Frodo",
            names=["Frodo"],
            summary="A hobbit",
            body="A hobbit who knows [gandalf](gandalf) and [aragorn](aragorn).",
            create=True,
        )

        # Should raise an error about broken links
        with pytest.raises(LLMSolvableError) as exc_info:
            tool._apply(block)

        error_msg = str(exc_info.value)
        assert "Cannot create wiki page with broken links" in error_msg
        assert "gandalf" in error_msg
        assert "aragorn" in error_msg

        # Verify the page was NOT written
        from bookwiki.models import WikiPage

        page = WikiPage.read_page_at(cursor, "frodo", chapter.id)
        assert page is None


def test_update_page_with_broken_links(temp_db: SafeConnection) -> None:
    """Test updating a page to add broken wiki links raises error."""
    with temp_db.transaction_cursor() as cursor:
        # Create a chapter
        chapter = Chapter.add_chapter(cursor, 1, ["Chapter 1"], "Some content")
        conversation = Conversation.create(cursor)
        chapter.set_conversation_id(conversation)

        # First create a page with no links
        block1 = Block.create_text(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            role="assistant",
            text="Creating frodo",
        )
        block1.write_wiki_page(
            chapter_id=chapter.id,
            slug="frodo",
            title="Frodo",
            names=["Frodo"],
            summary="A hobbit",
            body="A hobbit.",
        )

        # Now try to update it to add broken links
        block = Block.create_tool_use(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            name="WriteWikiPage",
            use_id="write_1",
            params='{"slug": "frodo", '
            '"body": "A hobbit who knows [gandalf](gandalf) and [aragorn](aragorn).", '
            '"create": false}',
        )

        tool = WriteWikiPage(
            tool_id="write_1",
            tool_name="WriteWikiPage",
            slug="frodo",
            title=None,
            names=None,
            summary=None,
            body="A hobbit who knows [gandalf](gandalf) and [aragorn](aragorn).",
            create=False,
        )

        # Should raise an error about broken links
        with pytest.raises(LLMSolvableError) as exc_info:
            tool._apply(block)

        error_msg = str(exc_info.value)
        assert "Cannot update wiki page with broken links" in error_msg
        assert "gandalf" in error_msg
        assert "aragorn" in error_msg

        # Verify the page was NOT updated
        from bookwiki.models import WikiPage

        page = WikiPage.read_page_at(cursor, "frodo", chapter.id)
        assert page is not None
        assert page.body == "A hobbit."  # Original body unchanged


def test_write_page_with_mixed_valid_invalid_links(temp_db: SafeConnection) -> None:
    """Test writing a page with both valid and invalid links raises error."""
    with temp_db.transaction_cursor() as cursor:
        # Create a chapter
        chapter = Chapter.add_chapter(cursor, 1, ["Chapter 1"], "Some content")
        conversation = Conversation.create(cursor)
        chapter.set_conversation_id(conversation)

        # Create some valid pages
        block1 = Block.create_text(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            role="assistant",
            text="Creating gandalf",
        )
        block1.write_wiki_page(
            chapter_id=chapter.id,
            slug="gandalf",
            title="Gandalf",
            names=["Gandalf"],
            summary="A wizard",
            body="A wizard.",
        )

        block2 = Block.create_text(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            role="assistant",
            text="Creating frodo",
        )
        block2.write_wiki_page(
            chapter_id=chapter.id,
            slug="frodo",
            title="Frodo",
            names=["Frodo"],
            summary="A hobbit",
            body="A hobbit.",
        )

        # Try to create a page with mix of valid and invalid links
        block = Block.create_tool_use(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            name="WriteWikiPage",
            use_id="write_1",
            params='{"slug": "story", "title": "The Story", "names": ["The Story"], '
            '"summary": "A story about heroes", '
            '"body": "Features [gandalf](gandalf) and [frodo](frodo), '
            'but also [sauron](sauron) and [gollum](gollum).", '
            '"create": true}',
        )

        tool = WriteWikiPage(
            tool_id="write_1",
            tool_name="WriteWikiPage",
            slug="story",
            title="The Story",
            names=["The Story"],
            summary="A story about heroes",
            body="Features [gandalf](gandalf) and [frodo](frodo), "
            "but also [sauron](sauron) and [gollum](gollum).",
            create=True,
        )

        # Should raise error about the invalid links only
        with pytest.raises(LLMSolvableError) as exc_info:
            tool._apply(block)

        error_msg = str(exc_info.value)
        assert "Cannot create wiki page with broken links" in error_msg
        assert "sauron" in error_msg
        assert "gollum" in error_msg

        # Verify the page was NOT written
        from bookwiki.models import WikiPage

        page = WikiPage.read_page_at(cursor, "story", chapter.id)
        assert page is None


def test_write_page_with_self_reference(temp_db: SafeConnection) -> None:
    """Test writing a page that references itself."""
    with temp_db.transaction_cursor() as cursor:
        # Create a chapter
        chapter = Chapter.add_chapter(cursor, 1, ["Chapter 1"], "Some content")
        conversation = Conversation.create(cursor)
        chapter.set_conversation_id(conversation)

        # Create a page that references itself
        block = Block.create_tool_use(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            name="WriteWikiPage",
            use_id="write_1",
            params='{"slug": "gandalf", "title": "Gandalf", "names": ["Gandalf"], '
            '"summary": "A wizard known as Gandalf", '
            '"body": "[Gandalf](gandalf) is a wizard known as Gandalf.", '
            '"create": true}',
        )

        tool = WriteWikiPage(
            tool_id="write_1",
            tool_name="WriteWikiPage",
            slug="gandalf",
            title="Gandalf",
            names=["Gandalf"],
            summary="A wizard known as Gandalf",
            body="[Gandalf](gandalf) is a wizard known as Gandalf.",
            create=True,
        )

        # Self-reference should fail since the page doesn't exist yet
        with pytest.raises(LLMSolvableError) as exc_info:
            tool._apply(block)

        error_msg = str(exc_info.value)
        assert "Cannot create wiki page with broken links" in error_msg
        assert "gandalf" in error_msg


def test_update_preserves_existing_body_links_check(temp_db: SafeConnection) -> None:
    """Test that updating non-body fields still validates existing body links."""
    with temp_db.transaction_cursor() as cursor:
        # Create a chapter
        chapter = Chapter.add_chapter(cursor, 1, ["Chapter 1"], "Some content")
        conversation = Conversation.create(cursor)
        chapter.set_conversation_id(conversation)

        # Create gandalf page
        block1 = Block.create_text(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            role="assistant",
            text="Creating gandalf",
        )
        block1.write_wiki_page(
            chapter_id=chapter.id,
            slug="gandalf",
            title="Gandalf",
            names=["Gandalf"],
            summary="A wizard",
            body="A wizard.",
        )

        # Create frodo page with link to gandalf
        block2 = Block.create_text(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            role="assistant",
            text="Creating frodo",
        )
        block2.write_wiki_page(
            chapter_id=chapter.id,
            slug="frodo",
            title="Frodo",
            names=["Frodo"],
            summary="A hobbit",
            body="A hobbit who knows [gandalf](gandalf).",
        )

        # Now effectively delete gandalf by overwriting with empty content
        block_delete = Block.create_text(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            role="assistant",
            text="Removing gandalf",
        )
        block_delete.write_wiki_page(
            chapter_id=chapter.id,
            slug="gandalf",
            title="",
            names=[""],
            summary="",
            body="",
        )

        # Try to update just the title of frodo
        # Should fail because existing body has broken link
        block = Block.create_tool_use(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            name="WriteWikiPage",
            use_id="write_1",
            params='{"slug": "frodo", "title": "Frodo Baggins", "create": false}',
        )

        tool = WriteWikiPage(
            tool_id="write_1",
            tool_name="WriteWikiPage",
            slug="frodo",
            title="Frodo Baggins",
            names=None,
            summary=None,
            body=None,
            create=False,
        )

        # Should raise error about broken links in existing body
        with pytest.raises(LLMSolvableError) as exc_info:
            tool._apply(block)

        error_msg = str(exc_info.value)
        assert "Cannot update wiki page with broken links" in error_msg
        assert "gandalf" in error_msg
