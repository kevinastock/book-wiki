"""Tests for WriteWikiPage tool with blank/empty values for removing redundant pages."""

from bookwiki.db import SafeConnection
from bookwiki.models import Block, Chapter, Conversation, WikiPage
from bookwiki.tools.wiki import WriteWikiPage


def test_write_wiki_page_with_blank_values_preserves_existing(
    temp_db: SafeConnection,
) -> None:
    """Test WriteWikiPage tool with empty/blank values preserves existing content."""
    with temp_db.transaction_cursor() as cursor:
        # Set up prerequisites
        chapter = Chapter.add_chapter(
            cursor, 1, ["Book 1", "Chapter 1"], "Chapter content..."
        )
        conversation = Conversation.create(cursor)
        chapter.set_conversation_id(conversation)  # Start the chapter

        # Create initial block and page
        create_block = Block.create_text(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            role="assistant",
            text="Creating redundant page",
        )

        # Create a wiki page that we later want to "remove" by blanking
        create_block.write_wiki_page(
            chapter_id=1,
            slug="redundant-page",
            title="Redundant Character",
            names=["Redundant Character", "Duplicate"],
            summary="This is a duplicate page that should be removed",
            body="This character page is redundant and should be blanked out.",
        )

        # Verify the page was created
        retrieved_page = WikiPage.read_page_at(cursor, "redundant-page", 1)
        assert retrieved_page is not None
        assert retrieved_page.title == "Redundant Character"
        assert "duplicate" in retrieved_page.summary.lower()
        assert len(retrieved_page.body) > 0

        # Now create another block to "blank out" the page
        blank_block = Block.create_text(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            role="assistant",
            text="Blanking redundant page",
        )

        # Use WriteWikiPage tool to blank out the page
        blank_tool = WriteWikiPage(
            tool_id="blank_tool_1",
            tool_name="WriteWikiPage",
            slug="redundant-page",
            title="",  # Empty title - falsy, should preserve existing
            names=[],  # Empty names list - falsy, should preserve existing
            summary="",  # Empty summary - falsy, should preserve existing
            body="",  # Empty body - falsy, should preserve existing
            create=False,
        )

        # This should work and preserve existing content (due to falsy empty values)
        blank_tool._apply(blank_block)

        # Verify existing content was preserved (due to falsy empty values)
        retrieved_page = WikiPage.read_page_at(cursor, "redundant-page", 1)
        assert retrieved_page is not None
        assert retrieved_page.title == "Redundant Character"
        assert set(retrieved_page.names) == {"Redundant Character", "Duplicate"}
        assert (
            retrieved_page.summary == "This is a duplicate page that should be removed"
        )
        assert (
            retrieved_page.body
            == "This character page is redundant and should be blanked out."
        )


def test_write_wiki_page_blank_values_minimal_content(temp_db: SafeConnection) -> None:
    """Test blanking a page with minimal valid content instead of completely empty."""
    with temp_db.transaction_cursor() as cursor:
        # Set up prerequisites
        chapter = Chapter.add_chapter(
            cursor, 1, ["Book 1", "Chapter 1"], "Chapter content..."
        )
        conversation = Conversation.create(cursor)
        chapter.set_conversation_id(conversation)  # Start the chapter

        # Create initial block and page
        create_block = Block.create_text(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            role="assistant",
            text="Creating page to be blanked",
        )

        # Create a wiki page
        create_block.write_wiki_page(
            chapter_id=1,
            slug="page-to-blank",
            title="Character to Remove",
            names=["Character to Remove", "Unwanted Character"],
            summary="This character page will be minimized",
            body="This character has detailed information that should be removed.",
        )

        # Create another block to blank the page with minimal content
        blank_block = Block.create_text(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            role="assistant",
            text="Blanking page with minimal content",
        )

        # Use WriteWikiPage tool to minimize the page content with whitespace (truthy)
        blank_tool = WriteWikiPage(
            tool_id="blank_tool_2",
            tool_name="WriteWikiPage",
            slug="page-to-blank",
            title=" ",  # Whitespace title (truthy, will replace)
            names=[" "],  # Whitespace name (truthy, will replace)
            summary=" ",  # Whitespace summary (truthy, will replace)
            body=" ",  # Whitespace body (truthy, will replace)
            create=False,
        )

        # This should work
        blank_tool._apply(blank_block)

        # Verify the page was "blanked" with whitespace content
        retrieved_page = WikiPage.read_page_at(cursor, "page-to-blank", 1)
        assert retrieved_page is not None
        assert retrieved_page.title == " "
        assert retrieved_page.names == [" "]
        assert retrieved_page.summary == " "
        assert retrieved_page.body == " "


def test_write_wiki_page_blank_update_preserves_existing_when_none_provided(
    temp_db: SafeConnection,
) -> None:
    """Test providing None/empty values in update mode preserves existing content."""
    with temp_db.transaction_cursor() as cursor:
        # Set up prerequisites
        chapter = Chapter.add_chapter(
            cursor, 1, ["Book 1", "Chapter 1"], "Chapter content..."
        )
        conversation = Conversation.create(cursor)
        chapter.set_conversation_id(conversation)  # Start the chapter

        # Create initial page
        create_block = Block.create_text(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            role="assistant",
            text="Creating page",
        )

        create_block.write_wiki_page(
            chapter_id=1,
            slug="test-preserve",
            title="Original Title",
            names=["Original Name", "Alias"],
            summary="Original summary",
            body="Original body content",
        )

        # Create update block
        update_block = Block.create_text(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            role="assistant",
            text="Updating page with None values",
        )

        # Update with None values - should preserve existing content
        update_tool = WriteWikiPage(
            tool_id="update_tool_1",
            tool_name="WriteWikiPage",
            slug="test-preserve",
            title=None,  # Should preserve existing title
            names=None,  # Should preserve existing names
            summary=None,  # Should preserve existing summary
            body=None,  # Should preserve existing body
            create=False,
        )

        update_tool._apply(update_block)

        # Verify existing content was preserved
        retrieved_page = WikiPage.read_page_at(cursor, "test-preserve", 1)
        assert retrieved_page is not None
        assert retrieved_page.title == "Original Title"
        assert set(retrieved_page.names) == {"Original Name", "Alias"}
        assert retrieved_page.summary == "Original summary"
        assert retrieved_page.body == "Original body content"


def test_write_wiki_page_blank_update_with_empty_strings_preserves_existing(
    temp_db: SafeConnection,
) -> None:
    """Test that providing empty strings in update mode preserves existing content."""
    with temp_db.transaction_cursor() as cursor:
        # Set up prerequisites
        chapter = Chapter.add_chapter(
            cursor, 1, ["Book 1", "Chapter 1"], "Chapter content..."
        )
        conversation = Conversation.create(cursor)
        chapter.set_conversation_id(conversation)  # Start the chapter

        # Create initial page
        create_block = Block.create_text(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            role="assistant",
            text="Creating page",
        )

        create_block.write_wiki_page(
            chapter_id=1,
            slug="test-empty-strings",
            title="Original Title",
            names=["Original Name"],
            summary="Original summary",
            body="Original body",
        )

        # Create update block
        update_block = Block.create_text(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            role="assistant",
            text="Updating with empty strings",
        )

        # Update with empty strings - should preserve existing due to falsy values
        update_tool = WriteWikiPage(
            tool_id="update_tool_2",
            tool_name="WriteWikiPage",
            slug="test-empty-strings",
            title="",  # Empty string is falsy - should preserve existing
            names=[],  # Empty list is falsy - should preserve existing
            summary="",  # Empty string is falsy - should preserve existing
            body="",  # Empty string is falsy - should preserve existing
            create=False,
        )

        update_tool._apply(update_block)

        # Verify existing content was preserved (due to falsy empty values)
        retrieved_page = WikiPage.read_page_at(cursor, "test-empty-strings", 1)
        assert retrieved_page is not None
        assert retrieved_page.title == "Original Title"
        assert retrieved_page.names == ["Original Name"]
        assert retrieved_page.summary == "Original summary"
        assert retrieved_page.body == "Original body"


def test_write_wiki_page_blank_update_with_whitespace_actually_updates(
    temp_db: SafeConnection,
) -> None:
    """Test whitespace-only strings actually updates (since they're truthy)."""
    with temp_db.transaction_cursor() as cursor:
        # Set up prerequisites
        chapter = Chapter.add_chapter(
            cursor, 1, ["Book 1", "Chapter 1"], "Chapter content..."
        )
        conversation = Conversation.create(cursor)
        chapter.set_conversation_id(conversation)  # Start the chapter

        # Create initial page
        create_block = Block.create_text(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            role="assistant",
            text="Creating page",
        )

        create_block.write_wiki_page(
            chapter_id=1,
            slug="test-whitespace",
            title="Original Title",
            names=["Original Name"],
            summary="Original summary",
            body="Original body",
        )

        # Create update block
        update_block = Block.create_text(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            role="assistant",
            text="Updating with whitespace",
        )

        # Update with whitespace strings - these are truthy so they should replace
        update_tool = WriteWikiPage(
            tool_id="update_tool_3",
            tool_name="WriteWikiPage",
            slug="test-whitespace",
            title=" ",  # Space is truthy - should replace
            names=[" "],  # List with space is truthy - should replace
            summary=" ",  # Space is truthy - should replace
            body=" ",  # Space is truthy - should replace
            create=False,
        )

        update_tool._apply(update_block)

        # Verify content was actually replaced with whitespace
        retrieved_page = WikiPage.read_page_at(cursor, "test-whitespace", 1)
        assert retrieved_page is not None
        assert retrieved_page.title == " "
        assert retrieved_page.names == [" "]
        assert retrieved_page.summary == " "
        assert retrieved_page.body == " "


def test_write_wiki_page_actual_removal_with_empty_content(
    temp_db: SafeConnection,
) -> None:
    """Test manually removing a page by overwriting with empty content."""
    with temp_db.transaction_cursor() as cursor:
        # Set up prerequisites
        chapter = Chapter.add_chapter(
            cursor, 1, ["Book 1", "Chapter 1"], "Chapter content..."
        )
        conversation = Conversation.create(cursor)
        chapter.set_conversation_id(conversation)  # Start the chapter

        # Create initial page
        create_block = Block.create_text(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            role="assistant",
            text="Creating page to remove",
        )

        create_block.write_wiki_page(
            chapter_id=1,
            slug="to-remove",
            title="Character to Remove",
            names=["Character to Remove", "Redundant Character"],
            summary="This character will be removed",
            body="This is a detailed character description that should be removed.",
        )

        # Create removal block
        remove_block = Block.create_text(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            role="assistant",
            text="Removing redundant page",
        )

        # Use WriteWikiPage tool to manually remove the page with whitespace (truthy)
        remove_tool = WriteWikiPage(
            tool_id="remove_tool",
            tool_name="WriteWikiPage",
            slug="to-remove",
            title=" ",  # Whitespace content (truthy, will replace)
            names=[" "],  # Whitespace name (truthy, will replace)
            summary=" ",  # Whitespace content (truthy, will replace)
            body=" ",  # Whitespace content (truthy, will replace)
            create=False,
        )

        remove_tool._apply(remove_block)

        # Verify the page was effectively "removed" (replaced with whitespace)
        retrieved_page = WikiPage.read_page_at(cursor, "to-remove", 1)
        assert retrieved_page is not None
        assert retrieved_page.title == " "
        assert retrieved_page.names == [" "]
        assert retrieved_page.summary == " "
        assert retrieved_page.body == " "


def test_write_wiki_page_delete_and_redirect_parameter(temp_db: SafeConnection) -> None:
    """Test the new delete_and_redirect_to parameter for page removal with links."""
    with temp_db.transaction_cursor() as cursor:
        # Set up prerequisites
        chapter = Chapter.add_chapter(
            cursor, 1, ["Book 1", "Chapter 1"], "Chapter content..."
        )
        conversation = Conversation.create(cursor)
        chapter.set_conversation_id(conversation)  # Start the chapter

        # Create target page for redirection
        target_block = Block.create_text(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            role="assistant",
            text="Creating target page",
        )

        target_block.write_wiki_page(
            chapter_id=1,
            slug="main-page",
            title="Main Page",
            names=["Main Page"],
            summary="The canonical page",
            body="This is the main page.",
        )

        # Create initial page to delete
        create_block = Block.create_text(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            role="assistant",
            text="Creating page to delete",
        )

        create_block.write_wiki_page(
            chapter_id=1,
            slug="to-delete",
            title="Redundant Page",
            names=["Redundant Page", "Duplicate"],
            summary="This page is redundant",
            body="This page should be deleted.",
        )

        # Create page with link to the page we're going to delete
        linking_block = Block.create_text(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            role="assistant",
            text="Creating page with link",
        )

        linking_block.write_wiki_page(
            chapter_id=1,
            slug="linking-page",
            title="Linking Page",
            names=["Linking Page"],
            summary="Page with a link",
            body="This links to [the redundant page](to-delete).",
        )

        # Create delete block
        delete_block = Block.create_text(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            role="assistant",
            text="Deleting redundant page",
        )

        # Use delete_and_redirect_to parameter to remove the page
        delete_tool = WriteWikiPage(
            tool_id="delete_tool",
            tool_name="WriteWikiPage",
            slug="to-delete",
            title=None,
            names=None,
            summary=None,
            body=None,
            create=False,
            delete_and_redirect_to="main-page",
        )

        delete_tool._apply(delete_block)

        # Verify the page was deleted (no longer retrievable)
        retrieved_page = WikiPage.read_page_at(cursor, "to-delete", 1)
        assert retrieved_page is None

        # Verify the linking page was updated
        linking_page = WikiPage.read_page_at(cursor, "linking-page", 1)
        assert linking_page is not None
        assert "This links to [the redundant page](main-page)." in linking_page.body

        # Verify the page still has 2 versions in history (original + deletion marker)
        all_versions = WikiPage.get_versions_by_slug(cursor, "to-delete", 1)
        assert len(all_versions) == 2  # Original + deletion marker


def test_write_wiki_page_delete_nonexistent_fails(temp_db: SafeConnection) -> None:
    """Test that trying to delete a non-existent page fails."""
    with temp_db.transaction_cursor() as cursor:
        # Set up prerequisites
        chapter = Chapter.add_chapter(
            cursor, 1, ["Book 1", "Chapter 1"], "Chapter content..."
        )
        conversation = Conversation.create(cursor)
        chapter.set_conversation_id(conversation)  # Start the chapter

        # Create a target page for redirection
        target_block = Block.create_text(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            role="assistant",
            text="Creating target page",
        )

        target_block.write_wiki_page(
            chapter_id=1,
            slug="target",
            title="Target Page",
            names=["Target Page"],
            summary="Target for redirection",
            body="This is the target page.",
        )

        # Create delete block
        delete_block = Block.create_text(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            role="assistant",
            text="Trying to delete non-existent page",
        )

        # Try to delete a page that doesn't exist
        delete_tool = WriteWikiPage(
            tool_id="delete_fail_tool",
            tool_name="WriteWikiPage",
            slug="nonexistent",
            title=None,
            names=None,
            summary=None,
            body=None,
            create=False,
            delete_and_redirect_to="target",
        )

        # This should handle the error and record it as a tool response
        delete_tool.apply(delete_block)  # Use apply() not _apply() for error handling

        # Re-read the block from database to get updated values
        updated_block = Block.get_by_id(cursor, delete_block.id)
        assert updated_block is not None

        # Check that an error response was recorded
        assert updated_block.tool_response is not None
        assert "Cannot delete non-existent page" in updated_block.tool_response
        assert updated_block.errored is True


def test_write_wiki_page_delete_wrong_value_ignored(temp_db: SafeConnection) -> None:
    """Test that normal updates work when delete_and_redirect_to is None."""
    with temp_db.transaction_cursor() as cursor:
        # Set up prerequisites
        chapter = Chapter.add_chapter(
            cursor, 1, ["Book 1", "Chapter 1"], "Chapter content..."
        )
        conversation = Conversation.create(cursor)
        chapter.set_conversation_id(conversation)  # Start the chapter

        # Create initial page
        create_block = Block.create_text(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            role="assistant",
            text="Creating page",
        )

        create_block.write_wiki_page(
            chapter_id=1,
            slug="test-page",
            title="Test Page",
            names=["Test"],
            summary="Test page",
            body="Test content",
        )

        # Create update block
        update_block = Block.create_text(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            role="assistant",
            text="Updating page with wrong delete value",
        )

        # Use None for delete_and_redirect_to - should work as normal update
        update_tool = WriteWikiPage(
            tool_id="wrong_delete_tool",
            tool_name="WriteWikiPage",
            slug="test-page",
            title="Updated Title",
            names=None,  # Should preserve existing
            summary=None,  # Should preserve existing
            body=None,  # Should preserve existing
            create=False,
            delete_and_redirect_to=None,  # No deletion requested
        )

        update_tool._apply(update_block)

        # Verify normal update happened (not deletion)
        retrieved_page = WikiPage.read_page_at(cursor, "test-page", 1)
        assert retrieved_page is not None
        assert retrieved_page.title == "Updated Title"  # Should be updated
        assert retrieved_page.names == ["Test"]  # Should be preserved
        assert retrieved_page.summary == "Test page"  # Should be preserved
        assert retrieved_page.body == "Test content"  # Should be preserved


def test_write_wiki_page_delete_with_content_fails(temp_db: SafeConnection) -> None:
    """Test that delete_and_redirect_to with content fields set fails validation."""
    with temp_db.transaction_cursor() as cursor:
        # Set up prerequisites
        chapter = Chapter.add_chapter(
            cursor, 1, ["Book 1", "Chapter 1"], "Chapter content..."
        )
        conversation = Conversation.create(cursor)
        chapter.set_conversation_id(conversation)  # Start the chapter

        # Create target page for redirection
        target_block = Block.create_text(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            role="assistant",
            text="Creating target page",
        )

        target_block.write_wiki_page(
            chapter_id=1,
            slug="target-page",
            title="Target Page",
            names=["Target Page"],
            summary="Target page",
            body="Target content",
        )

        # Create initial page
        create_block = Block.create_text(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            role="assistant",
            text="Creating page",
        )

        create_block.write_wiki_page(
            chapter_id=1,
            slug="test-delete-validation",
            title="Test Page",
            names=["Test"],
            summary="Test page",
            body="Test content",
        )

        # Create delete block
        delete_block = Block.create_text(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            role="assistant",
            text="Attempting invalid delete with content",
        )

        # Try to delete with content fields set - should fail validation
        delete_tool = WriteWikiPage(
            tool_id="invalid_delete_tool",
            tool_name="WriteWikiPage",
            slug="test-delete-validation",
            title="Some Title",  # This should cause validation error
            names=None,
            summary=None,
            body=None,
            create=False,
            delete_and_redirect_to="target-page",
        )

        # This should handle the validation error
        delete_tool.apply(delete_block)

        # Re-read the block from database to get updated values
        updated_block = Block.get_by_id(cursor, delete_block.id)
        assert updated_block is not None

        # Check that a validation error response was recorded
        assert updated_block.tool_response is not None
        assert "Cannot set content fields" in updated_block.tool_response
        assert "when using delete_and_redirect_to" in updated_block.tool_response
        assert updated_block.errored is True

        # Verify the original page was not modified
        retrieved_page = WikiPage.read_page_at(cursor, "test-delete-validation", 1)
        assert retrieved_page is not None
        assert retrieved_page.title == "Test Page"  # Should be unchanged
        assert retrieved_page.names == ["Test"]
        assert retrieved_page.summary == "Test page"
        assert retrieved_page.body == "Test content"


def test_write_wiki_page_delete_with_names_content_fails(
    temp_db: SafeConnection,
) -> None:
    """Test that delete_and_redirect_to with names content fails validation."""
    with temp_db.transaction_cursor() as cursor:
        # Set up prerequisites
        chapter = Chapter.add_chapter(
            cursor, 1, ["Book 1", "Chapter 1"], "Chapter content..."
        )
        conversation = Conversation.create(cursor)
        chapter.set_conversation_id(conversation)  # Start the chapter

        # Create target page for redirection
        target_block = Block.create_text(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            role="assistant",
            text="Creating target page",
        )

        target_block.write_wiki_page(
            chapter_id=1,
            slug="target-page",
            title="Target Page",
            names=["Target Page"],
            summary="Target page",
            body="Target content",
        )

        # Create initial page
        create_block = Block.create_text(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            role="assistant",
            text="Creating page",
        )

        create_block.write_wiki_page(
            chapter_id=1,
            slug="test-delete-names",
            title="Test Page",
            names=["Test"],
            summary="Test page",
            body="Test content",
        )

        # Create delete block
        delete_block = Block.create_text(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            role="assistant",
            text="Attempting invalid delete with names",
        )

        # Try to delete with names field set - should fail validation
        delete_tool = WriteWikiPage(
            tool_id="invalid_delete_names_tool",
            tool_name="WriteWikiPage",
            slug="test-delete-names",
            title=None,
            names=["Some Name"],  # This should cause validation error
            summary=None,
            body=None,
            create=False,
            delete_and_redirect_to="target-page",
        )

        # This should handle the validation error
        delete_tool.apply(delete_block)

        # Re-read the block from database to get updated values
        updated_block = Block.get_by_id(cursor, delete_block.id)
        assert updated_block is not None

        # Check that a validation error response was recorded
        assert updated_block.tool_response is not None
        assert "Cannot set content fields" in updated_block.tool_response
        assert updated_block.errored is True

        # Verify the original page was not modified (names validation failed)
        retrieved_page = WikiPage.read_page_at(cursor, "test-delete-names", 1)
        assert retrieved_page is not None
        assert retrieved_page.title == "Test Page"  # Should be unchanged
        assert retrieved_page.names == ["Test"]
        assert retrieved_page.summary == "Test page"
        assert retrieved_page.body == "Test content"


def test_write_wiki_page_delete_self_redirect_fails(temp_db: SafeConnection) -> None:
    """Test that trying to redirect a page to itself fails."""
    with temp_db.transaction_cursor() as cursor:
        # Set up prerequisites
        chapter = Chapter.add_chapter(
            cursor, 1, ["Book 1", "Chapter 1"], "Chapter content..."
        )
        conversation = Conversation.create(cursor)
        chapter.set_conversation_id(conversation)  # Start the chapter

        # Create initial page
        create_block = Block.create_text(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            role="assistant",
            text="Creating page",
        )

        create_block.write_wiki_page(
            chapter_id=1,
            slug="self-test",
            title="Self Test Page",
            names=["Self Test"],
            summary="Page for self-redirect test",
            body="This page will try to redirect to itself.",
        )

        # Create delete block
        delete_block = Block.create_text(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            role="assistant",
            text="Attempting self-redirect",
        )

        # Try to redirect page to itself
        delete_tool = WriteWikiPage(
            tool_id="self_redirect_tool",
            tool_name="WriteWikiPage",
            slug="self-test",
            title=None,
            names=None,
            summary=None,
            body=None,
            create=False,
            delete_and_redirect_to="self-test",
        )

        # This should handle the validation error
        delete_tool.apply(delete_block)

        # Re-read the block from database to get updated values
        updated_block = Block.get_by_id(cursor, delete_block.id)
        assert updated_block is not None

        # Check that a validation error response was recorded
        assert updated_block.tool_response is not None
        assert "Cannot redirect a page to itself" in updated_block.tool_response
        assert updated_block.errored is True

        # Verify the original page was not modified (self-redirect failed)
        retrieved_page = WikiPage.read_page_at(cursor, "self-test", 1)
        assert retrieved_page is not None
        assert retrieved_page.title == "Self Test Page"  # Should be unchanged


def test_write_wiki_page_delete_with_empty_string_removes_links(
    temp_db: SafeConnection,
) -> None:
    """Test that delete_and_redirect_to with empty string removes links."""
    with temp_db.transaction_cursor() as cursor:
        # Set up prerequisites
        chapter = Chapter.add_chapter(
            cursor, 1, ["Book 1", "Chapter 1"], "Chapter content..."
        )
        conversation = Conversation.create(cursor)
        chapter.set_conversation_id(conversation)  # Start the chapter

        # Create initial page to delete
        create_block = Block.create_text(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            role="assistant",
            text="Creating page to delete",
        )

        create_block.write_wiki_page(
            chapter_id=1,
            slug="to-delete",
            title="Page to Delete",
            names=["Page to Delete"],
            summary="This page will be deleted",
            body="This page should be deleted with links removed.",
        )

        # Create page with link to the page we're going to delete
        linking_block = Block.create_text(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            role="assistant",
            text="Creating page with link",
        )

        linking_block.write_wiki_page(
            chapter_id=1,
            slug="linking-page",
            title="Linking Page",
            names=["Linking Page"],
            summary="Page with a link",
            body="This links to [the page to delete](to-delete) and mentions it again.",
        )

        # Create another page with multiple links to the page to delete
        multi_link_block = Block.create_text(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            role="assistant",
            text="Creating page with multiple links",
        )

        multi_link_block.write_wiki_page(
            chapter_id=1,
            slug="multi-link-page",
            title="Multi Link Page",
            names=["Multi Link Page"],
            summary="Page with multiple links",
            body="Here's [one link](to-delete) and [another link](to-delete).",
        )

        # Create delete block
        delete_block = Block.create_text(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            role="assistant",
            text="Deleting page and removing links",
        )

        # Use delete_and_redirect_to with empty string to remove links
        delete_tool = WriteWikiPage(
            tool_id="delete_and_unlink_tool",
            tool_name="WriteWikiPage",
            slug="to-delete",
            title=None,
            names=None,
            summary=None,
            body=None,
            create=False,
            delete_and_redirect_to="",  # Empty string should remove links
        )

        delete_tool._apply(delete_block)

        # Verify the page was deleted (no longer retrievable)
        retrieved_page = WikiPage.read_page_at(cursor, "to-delete", 1)
        assert retrieved_page is None

        # Verify the linking page had its link removed (converted to plain text)
        linking_page = WikiPage.read_page_at(cursor, "linking-page", 1)
        assert linking_page is not None
        assert (
            "This links to the page to delete and mentions it again."
            in linking_page.body
        )
        assert "[the page to delete](to-delete)" not in linking_page.body

        # Verify the multi-link page had all its links removed
        multi_link_page = WikiPage.read_page_at(cursor, "multi-link-page", 1)
        assert multi_link_page is not None
        assert "Here's one link and another link." in multi_link_page.body
        assert "[one link](to-delete)" not in multi_link_page.body
        assert "[another link](to-delete)" not in multi_link_page.body


def test_write_wiki_page_delete_with_empty_string_nonexistent_redirect_target(
    temp_db: SafeConnection,
) -> None:
    """Test that delete_and_redirect_to with empty string works without validation."""
    with temp_db.transaction_cursor() as cursor:
        # Set up prerequisites
        chapter = Chapter.add_chapter(
            cursor, 1, ["Book 1", "Chapter 1"], "Chapter content..."
        )
        conversation = Conversation.create(cursor)
        chapter.set_conversation_id(conversation)  # Start the chapter

        # Create initial page to delete
        create_block = Block.create_text(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            role="assistant",
            text="Creating page to delete",
        )

        create_block.write_wiki_page(
            chapter_id=1,
            slug="to-unlink",
            title="Page to Unlink",
            names=["Page to Unlink"],
            summary="This page will be deleted",
            body="This page should be deleted with links removed.",
        )

        # Create page with link to the page we're going to delete
        linking_block = Block.create_text(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            role="assistant",
            text="Creating page with link",
        )

        linking_block.write_wiki_page(
            chapter_id=1,
            slug="linking-page-2",
            title="Linking Page 2",
            names=["Linking Page 2"],
            summary="Page with a link",
            body="This page links to [the page](to-unlink) that will be removed.",
        )

        # Create delete block
        delete_block = Block.create_text(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            role="assistant",
            text="Deleting page and removing links",
        )

        # Use delete_and_redirect_to with empty string - should work without validation
        delete_tool = WriteWikiPage(
            tool_id="unlink_tool",
            tool_name="WriteWikiPage",
            slug="to-unlink",
            title=None,
            names=None,
            summary=None,
            body=None,
            create=False,
            delete_and_redirect_to="",  # Empty string should skip target validation
        )

        delete_tool._apply(delete_block)

        # Verify the page was deleted
        retrieved_page = WikiPage.read_page_at(cursor, "to-unlink", 1)
        assert retrieved_page is None

        # Verify the linking page had its link removed
        linking_page = WikiPage.read_page_at(cursor, "linking-page-2", 1)
        assert linking_page is not None
        assert "This page links to the page that will be removed." in linking_page.body
        assert "[the page](to-unlink)" not in linking_page.body


def test_wikipage_delete_and_redirect_method_with_target(
    temp_db: SafeConnection,
) -> None:
    """Test the WikiPage.delete_and_redirect method with a valid redirect target."""
    with temp_db.transaction_cursor() as cursor:
        # Set up prerequisites
        chapter = Chapter.add_chapter(
            cursor, 1, ["Book 1", "Chapter 1"], "Chapter content..."
        )
        conversation = Conversation.create(cursor)
        chapter.set_conversation_id(conversation)

        # Create target page
        target_block = Block.create_text(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            role="assistant",
            text="Creating target page",
        )

        target_block.write_wiki_page(
            chapter_id=1,
            slug="target-page",
            title="Target Page",
            names=["Target Page"],
            summary="The target page",
            body="This is the target page for redirection.",
        )

        # Create page to delete
        delete_page_block = Block.create_text(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            role="assistant",
            text="Creating page to delete",
        )

        delete_page_block.write_wiki_page(
            chapter_id=1,
            slug="source-page",
            title="Source Page",
            names=["Source Page"],
            summary="The page to delete",
            body="This page will be deleted.",
        )

        # Create page with links to the page we'll delete
        linking_block = Block.create_text(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            role="assistant",
            text="Creating linking page",
        )

        linking_block.write_wiki_page(
            chapter_id=1,
            slug="linking-page-3",
            title="Linking Page 3",
            names=["Linking Page 3"],
            summary="Page with links",
            body="This links to [the source page](source-page) and mentions it.",
        )

        # Get the page to delete
        page_to_delete = WikiPage.read_page_at(cursor, "source-page", 1)
        assert page_to_delete is not None

        # Create a block for writing updates
        action_block = Block.create_text(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            role="assistant",
            text="Performing deletion",
        )

        # Call the delete_and_redirect method directly
        pages_updated, response = page_to_delete.delete_and_redirect(
            action_block, "target-page"
        )

        # Verify the response
        assert (
            "Wiki page 'source-page' deleted and redirected to 'target-page'."
            in response
        )
        assert "Updated 1 page(s) with redirected links." in response
        assert len(pages_updated) == 1
        assert pages_updated[0].slug == "linking-page-3"

        # Verify the source page is no longer accessible
        deleted_page = WikiPage.read_page_at(cursor, "source-page", 1)
        assert deleted_page is None

        # Verify the linking page was updated
        updated_linking_page = WikiPage.read_page_at(cursor, "linking-page-3", 1)
        assert updated_linking_page is not None
        assert (
            "This links to [the source page](target-page) and mentions it."
            in updated_linking_page.body
        )
        assert "[the source page](source-page)" not in updated_linking_page.body


def test_wikipage_delete_and_redirect_method_with_empty_string(
    temp_db: SafeConnection,
) -> None:
    """Test the WikiPage.delete_and_redirect method with empty string."""
    with temp_db.transaction_cursor() as cursor:
        # Set up prerequisites
        chapter = Chapter.add_chapter(
            cursor, 1, ["Book 1", "Chapter 1"], "Chapter content..."
        )
        conversation = Conversation.create(cursor)
        chapter.set_conversation_id(conversation)

        # Create page to delete
        delete_page_block = Block.create_text(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            role="assistant",
            text="Creating page to delete",
        )

        delete_page_block.write_wiki_page(
            chapter_id=1,
            slug="page-to-remove",
            title="Page to Remove",
            names=["Page to Remove"],
            summary="Page that will be removed",
            body="This page will be completely removed.",
        )

        # Create page with links to the page we'll delete
        linking_block = Block.create_text(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            role="assistant",
            text="Creating linking page",
        )

        linking_block.write_wiki_page(
            chapter_id=1,
            slug="linking-page-4",
            title="Linking Page 4",
            names=["Linking Page 4"],
            summary="Page with links to remove",
            body="Reference to [the removed page](page-to-remove) here.",
        )

        # Get the page to delete
        page_to_delete = WikiPage.read_page_at(cursor, "page-to-remove", 1)
        assert page_to_delete is not None

        # Create a block for writing updates
        action_block = Block.create_text(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            role="assistant",
            text="Performing deletion",
        )

        # Call the delete_and_redirect method with empty string
        pages_updated, response = page_to_delete.delete_and_redirect(action_block, "")

        # Verify the response
        assert "Wiki page 'page-to-remove' deleted and all links removed." in response
        assert "Updated 1 page(s) with redirected links." in response
        assert len(pages_updated) == 1
        assert pages_updated[0].slug == "linking-page-4"

        # Verify the source page is no longer accessible
        deleted_page = WikiPage.read_page_at(cursor, "page-to-remove", 1)
        assert deleted_page is None

        # Verify the linking page had its link removed (converted to plain text)
        updated_linking_page = WikiPage.read_page_at(cursor, "linking-page-4", 1)
        assert updated_linking_page is not None
        assert "Reference to the removed page here." in updated_linking_page.body
        assert "[the removed page](page-to-remove)" not in updated_linking_page.body


def test_wikipage_delete_and_redirect_method_no_links_to_update(
    temp_db: SafeConnection,
) -> None:
    """Test the WikiPage.delete_and_redirect method when no other pages link to it."""
    with temp_db.transaction_cursor() as cursor:
        # Set up prerequisites
        chapter = Chapter.add_chapter(
            cursor, 1, ["Book 1", "Chapter 1"], "Chapter content..."
        )
        conversation = Conversation.create(cursor)
        chapter.set_conversation_id(conversation)

        # Create page to delete (with no other pages linking to it)
        delete_page_block = Block.create_text(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            role="assistant",
            text="Creating isolated page",
        )

        delete_page_block.write_wiki_page(
            chapter_id=1,
            slug="isolated-page",
            title="Isolated Page",
            names=["Isolated Page"],
            summary="Page with no incoming links",
            body="This page has no incoming links from other pages.",
        )

        # Create another page that doesn't link to our target
        other_block = Block.create_text(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            role="assistant",
            text="Creating unrelated page",
        )

        other_block.write_wiki_page(
            chapter_id=1,
            slug="unrelated-page",
            title="Unrelated Page",
            names=["Unrelated Page"],
            summary="Unrelated page",
            body="This page doesn't link to the isolated page.",
        )

        # Get the page to delete
        page_to_delete = WikiPage.read_page_at(cursor, "isolated-page", 1)
        assert page_to_delete is not None

        # Create a block for writing updates
        action_block = Block.create_text(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            role="assistant",
            text="Performing deletion",
        )

        # Call the delete_and_redirect method
        pages_updated, response = page_to_delete.delete_and_redirect(action_block, "")

        # Verify the response (no pages were updated)
        assert "Wiki page 'isolated-page' deleted and all links removed." in response
        assert "Updated 0 page(s)" not in response  # Should not mention updated pages
        assert len(pages_updated) == 0

        # Verify the source page is no longer accessible
        deleted_page = WikiPage.read_page_at(cursor, "isolated-page", 1)
        assert deleted_page is None

        # Verify the unrelated page was not modified
        unrelated_page = WikiPage.read_page_at(cursor, "unrelated-page", 1)
        assert unrelated_page is not None
        assert unrelated_page.body == "This page doesn't link to the isolated page."


def test_wikipage_replace_links_with_path_prefix(temp_db: SafeConnection) -> None:
    """Test that link replacement preserves path prefixes in targets."""
    with temp_db.transaction_cursor() as cursor:
        # Set up prerequisites
        chapter = Chapter.add_chapter(
            cursor, 1, ["Book 1", "Chapter 1"], "Chapter content..."
        )
        conversation = Conversation.create(cursor)
        chapter.set_conversation_id(conversation)

        # Create target page
        target_block = Block.create_text(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            role="assistant",
            text="Creating target page",
        )

        target_block.write_wiki_page(
            chapter_id=1,
            slug="new-target",
            title="New Target",
            names=["New Target"],
            summary="The new target page",
            body="This is the new target.",
        )

        # Create page to delete
        delete_page_block = Block.create_text(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            role="assistant",
            text="Creating page to delete",
        )

        delete_page_block.write_wiki_page(
            chapter_id=1,
            slug="old-page",
            title="Old Page",
            names=["Old Page"],
            summary="Page to be deleted",
            body="This is the old page.",
        )

        # Create page with various link formats (including path prefixes)
        linking_block = Block.create_text(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            role="assistant",
            text="Creating page with various links",
        )

        linking_block.write_wiki_page(
            chapter_id=1,
            slug="complex-links",
            title="Complex Links Page",
            names=["Complex Links"],
            summary="Page with various link formats",
            body=(
                "Simple link: [old page](old-page)\n"
                "Path link: [old page with path](path/old-page)\n"
                "Deep path: [old page deep](deep/nested/old-page)\n"
                "Other link: [unrelated](other-page)"
            ),
        )

        # Get the page to delete
        page_to_delete = WikiPage.read_page_at(cursor, "old-page", 1)
        assert page_to_delete is not None

        # Create a block for writing updates
        action_block = Block.create_text(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            role="assistant",
            text="Performing deletion with path preservation",
        )

        # Call the delete_and_redirect method
        pages_updated, response = page_to_delete.delete_and_redirect(
            action_block, "new-target"
        )

        # Verify the linking page was updated with path prefixes preserved
        updated_page = WikiPage.read_page_at(cursor, "complex-links", 1)
        assert updated_page is not None
        expected_body = (
            "Simple link: [old page](new-target)\n"
            "Path link: [old page with path](path/new-target)\n"
            "Deep path: [old page deep](deep/nested/new-target)\n"
            "Other link: [unrelated](other-page)"
        )
        assert updated_page.body == expected_body


def test_wikipage_replace_links_with_empty_string_preserves_display_text(
    temp_db: SafeConnection,
) -> None:
    """Test that replacing links with empty string preserves only the display text."""
    with temp_db.transaction_cursor() as cursor:
        # Set up prerequisites
        chapter = Chapter.add_chapter(
            cursor, 1, ["Book 1", "Chapter 1"], "Chapter content..."
        )
        conversation = Conversation.create(cursor)
        chapter.set_conversation_id(conversation)

        # Create page to delete
        delete_page_block = Block.create_text(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            role="assistant",
            text="Creating page to delete",
        )

        delete_page_block.write_wiki_page(
            chapter_id=1,
            slug="page-to-unlink",
            title="Page to Unlink",
            names=["Page to Unlink"],
            summary="Page to be unlinked",
            body="This page will be deleted and unlinked.",
        )

        # Create page with various link formats and display texts
        linking_block = Block.create_text(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            role="assistant",
            text="Creating page with display texts",
        )

        linking_block.write_wiki_page(
            chapter_id=1,
            slug="display-text-links",
            title="Display Text Links",
            names=["Display Text Links"],
            summary="Page with various display texts",
            body=(
                "Here's [a nice description](page-to-unlink) of the page.\n"
                "Another [different text](page-to-unlink) for the same link.\n"
                "And [yet another description](path/page-to-unlink) with path.\n"
                "Normal [other link](different-page) should be unchanged."
            ),
        )

        # Get the page to delete
        page_to_delete = WikiPage.read_page_at(cursor, "page-to-unlink", 1)
        assert page_to_delete is not None

        # Create a block for writing updates
        action_block = Block.create_text(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            role="assistant",
            text="Unlinking page",
        )

        # Call the delete_and_redirect method with empty string
        pages_updated, response = page_to_delete.delete_and_redirect(action_block, "")

        # Verify the linking page had links replaced with display text only
        updated_page = WikiPage.read_page_at(cursor, "display-text-links", 1)
        assert updated_page is not None
        expected_body = (
            "Here's a nice description of the page.\n"
            "Another different text for the same link.\n"
            "And yet another description with path.\n"
            "Normal [other link](different-page) should be unchanged."
        )
        assert updated_page.body == expected_body


def test_write_wiki_page_delete_redirect_to_nonexistent_fails(
    temp_db: SafeConnection,
) -> None:
    """Test that delete_and_redirect_to with nonexistent target fails."""
    with temp_db.transaction_cursor() as cursor:
        # Set up prerequisites
        chapter = Chapter.add_chapter(
            cursor, 1, ["Book 1", "Chapter 1"], "Chapter content..."
        )
        conversation = Conversation.create(cursor)
        chapter.set_conversation_id(conversation)  # Start the chapter

        # Create page to delete
        create_block = Block.create_text(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            role="assistant",
            text="Creating page to delete",
        )

        create_block.write_wiki_page(
            chapter_id=1,
            slug="page-to-delete",
            title="Page to Delete",
            names=["Page to Delete"],
            summary="This page will be deleted",
            body="This page will be deleted.",
        )

        # Create a similar page for suggestions
        similar_block = Block.create_text(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            role="assistant",
            text="Creating similar page",
        )

        similar_block.write_wiki_page(
            chapter_id=1,
            slug="similar-target",
            title="Similar Target",
            names=["Similar Target"],
            summary="Similar to the nonexistent target",
            body="This is a similar page.",
        )

        # Create delete block
        delete_block = Block.create_text(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            role="assistant",
            text="Attempting delete with nonexistent redirect",
        )

        # Try to redirect to a nonexistent page
        delete_tool = WriteWikiPage(
            tool_id="delete_nonexistent_tool",
            tool_name="WriteWikiPage",
            slug="page-to-delete",
            title=None,
            names=None,
            summary=None,
            body=None,
            create=False,
            delete_and_redirect_to="similar-target-typo",  # Nonexistent but similar
        )

        # This should handle the error and record it as a tool response
        delete_tool.apply(delete_block)

        # Re-read the block from database to get updated values
        updated_block = Block.get_by_id(cursor, delete_block.id)
        assert updated_block is not None

        # Check that an error response was recorded with suggestions
        assert updated_block.tool_response is not None
        assert (
            "Cannot redirect to non-existent page 'similar-target-typo'"
            in updated_block.tool_response
        )
        assert "Did you mean one of these?" in updated_block.tool_response
        assert "similar-target" in updated_block.tool_response
        assert updated_block.errored is True

        # Verify the original page was not deleted (error occurred)
        retrieved_page = WikiPage.read_page_at(cursor, "page-to-delete", 1)
        assert retrieved_page is not None
        assert retrieved_page.title == "Page to Delete"
