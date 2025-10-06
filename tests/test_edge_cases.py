"""Edge case tests for bookwiki - unicode handling and malformed data."""

import threading
import time

import pytest

from bookwiki.db import SafeConnection
from bookwiki.models import Chapter, Conversation, WikiPage


def test_unicode_content_handling(temp_db: SafeConnection) -> None:
    """Test handling of various unicode characters in content."""
    with temp_db.transaction_cursor() as cursor:
        # Test unicode in chapter names and content
        Chapter.add_chapter(
            cursor,
            chapter_id=1,
            name=["魔法の世界", "第一章", "始まり 🌟"],
            text="昔々、遠い国に住んでいた魔法使いの物語です。\n彼は🧙‍♂️で、とても強力な魔法を使えました。",
        )

        # Verify chapter was stored and retrieved correctly
        retrieved_chapter = Chapter.read_chapter(cursor, 1)
        assert retrieved_chapter is not None
        assert retrieved_chapter.name == ["魔法の世界", "第一章", "始まり 🌟"]
        assert "魔法使い" in retrieved_chapter.text
        assert "🧙‍♂️" in retrieved_chapter.text

        # Test unicode in conversation text
        conversation = Conversation.create(cursor)
        conversation.add_user_text("こんにちは！第一章を読んでください。Émojis: 📖🌍✨")
        conversation.add_assistant_text("はい！読んでみます。Ça va bien! 🎭🎨🎪")

        # Verify text storage and retrieval
        blocks = conversation.blocks
        user_block = next(b for b in blocks if b.text_role == "user")
        assistant_block = next(b for b in blocks if b.text_role == "assistant")

        assert user_block.text_body is not None
        assert "こんにちは" in user_block.text_body
        assert "📖🌍✨" in user_block.text_body
        assert assistant_block.text_body is not None
        assert "はい" in assistant_block.text_body
        assert "Ça va bien" in assistant_block.text_body
        assert "🎭🎨🎪" in assistant_block.text_body

        # Test unicode in wiki page
        create_block = conversation.add_assistant_text("ウィキページを作成します")
        create_block.write_wiki_page(
            chapter_id=1,
            slug="unicode-character",
            title="魔法使いキャラクター",
            names=["魔法使い", "Wizard 🧙‍♂️", "Магик", "巫师"],
            summary="多言語の魔法使いキャラクター with émojis 🌟",
            body="このキャラクターは様々な言語で知られています:\n"
            "- 日本語: 魔法使い\n"
            "- English: Wizard 🧙‍♂️\n"
            "- Русский: Магик\n"
            "- 中文: 巫师\n"
            "Special chars: áéíóú àèìòù âêîôû ñç",
        )

        # Verify wiki page unicode handling
        retrieved_wiki = WikiPage.read_page_at(cursor, "unicode-character", 1)
        assert retrieved_wiki is not None
        assert retrieved_wiki.title == "魔法使いキャラクター"
        # All four names should be preserved as they're distinct Unicode
        assert set(retrieved_wiki.names) == {"魔法使い", "Wizard 🧙‍♂️", "Магик", "巫师"}
        assert "émojis 🌟" in retrieved_wiki.summary
        assert "áéíóú àèìòù" in retrieved_wiki.body


def test_extreme_content_lengths(temp_db: SafeConnection) -> None:
    """Test handling of very long content strings."""
    with temp_db.transaction_cursor() as cursor:
        # Create very long text content
        long_text = "Lorem ipsum dolor sit amet. " * 10000  # ~280k chars
        very_long_name = "VeryLongCharacterNameThatGoesOnAndOnAndOn" * 100  # ~4k chars

        # Test long chapter content
        Chapter.add_chapter(
            cursor,
            chapter_id=1,
            name=["Book", "Very Long Chapter"],
            text=long_text,
        )

        retrieved_chapter = Chapter.read_chapter(cursor, 1)
        assert retrieved_chapter is not None
        assert len(retrieved_chapter.text) == len(long_text)
        assert retrieved_chapter.text == long_text

        # Test long conversation text
        conversation = Conversation.create(cursor)
        conversation.add_user_text(long_text[:1000])  # Reasonable limit

        blocks = conversation.blocks
        assert blocks[0].text_body is not None
        assert len(blocks[0].text_body) == 1000

        # Test long wiki page content
        create_block = conversation.add_assistant_text("Creating long page")
        create_block.write_wiki_page(
            chapter_id=1,
            slug="long-content",
            title="Long Content Page",
            names=["Long Page", very_long_name[:500]],  # Reasonable name limit
            summary="A page with very long content",
            body=long_text[:5000],  # Reasonable body limit
        )

        retrieved_wiki = WikiPage.read_page_at(cursor, "long-content", 1)
        assert retrieved_wiki is not None
        assert len(retrieved_wiki.body) == 5000


def test_special_characters_in_slugs_and_names(temp_db: SafeConnection) -> None:
    """Test handling of special characters in slugs and identifiers."""
    with temp_db.transaction_cursor() as cursor:
        Chapter.add_chapter(cursor, 1, ["Book"], "Content")
        conversation = Conversation.create(cursor)

        # Test special characters in slugs (should be handled by application logic)
        special_slug_tests = [
            ("simple-slug", "Simple Page"),
            ("slug-with-numbers-123", "Numbered Page"),
            ("slug_with_underscores", "Underscore Page"),
            ("slug.with.dots", "Dotted Page"),  # May need escaping
        ]

        for slug, title in special_slug_tests:
            create_block = conversation.add_assistant_text(f"Creating {title}")
            create_block.write_wiki_page(
                chapter_id=1,
                slug=slug,
                title=title,
                names=[title],
                summary=f"Test page for {slug}",
                body=f"Content for {slug}",
            )

            # Verify retrieval works
            retrieved = WikiPage.read_page_at(cursor, slug, 1)
            assert retrieved is not None
            assert retrieved.slug == slug
            assert retrieved.title == title

        # Test special characters in names (should be preserved)
        special_names = [
            "Name with spaces",
            "Name-with-dashes",
            "Name_with_underscores",
            "Name.with.dots",
            "Name (with parentheses)",
            "Name [with brackets]",
            "Name 'with quotes'",
            'Name "with double quotes"',
            "Name & with ampersand",
            "Name @ with at-symbol",
            "Name # with hash",
            "Name % with percent",
        ]

        special_block = conversation.add_assistant_text("Creating special names page")
        special_block.write_wiki_page(
            chapter_id=1,
            slug="special-names",
            title="Special Names Test",
            names=special_names,
            summary="Testing special characters in names",
            body="This page tests various special characters in names.",
        )

        retrieved_special = WikiPage.read_page_at(cursor, "special-names", 1)
        assert retrieved_special is not None

        # All special names should be preserved
        for special_name in special_names:
            assert special_name in retrieved_special.names


def test_null_and_empty_value_handling(temp_db: SafeConnection) -> None:
    """Test handling of null and empty values in various fields."""
    with temp_db.transaction_cursor() as cursor:
        Chapter.add_chapter(cursor, 1, ["Book"], "Content")
        conversation = Conversation.create(cursor)

        # Test empty string handling in text blocks
        conversation.add_user_text("")
        conversation.add_assistant_text("")

        blocks = conversation.blocks
        user_block = next(b for b in blocks if b.text_role == "user")
        assistant_block = next(b for b in blocks if b.text_role == "assistant")

        assert user_block.text_body == ""
        assert assistant_block.text_body == ""

        # Test that empty chapter names are properly rejected
        with pytest.raises(ValueError, match="Chapter name list cannot be empty"):
            Chapter.add_chapter(
                cursor,
                chapter_id=2,
                name=[],  # Empty name list
                text="Chapter with no name",
            )

        # Test minimal wiki page (empty optional fields for updates)
        create_block = conversation.add_assistant_text("Creating minimal page")

        # First create a page
        initial_wiki = create_block.write_wiki_page(
            chapter_id=1,
            slug="minimal-page",
            title="Initial Title",
            names=["Initial Name"],
            summary="Initial summary",
            body="Initial body",
        )

        # Test that page was created
        assert initial_wiki.title == "Initial Title"


def test_boundary_value_handling(temp_db: SafeConnection) -> None:
    """Test handling of boundary values like very large numbers."""
    with temp_db.transaction_cursor() as cursor:
        conversation = Conversation.create(cursor)

        # Test very large token counts
        large_input_tokens = 2**31 - 1  # Max 32-bit signed int
        large_output_tokens = 2**30

        conversation.update_tokens(large_input_tokens, large_output_tokens)

        # Verify large numbers are stored correctly using helper
        from tests.helpers import verify_conversation_token_state

        verify_conversation_token_state(
            cursor,
            conversation,
            large_input_tokens,
            large_output_tokens,
            large_input_tokens + large_output_tokens,
        )

        # Test negative chapter indices (should be handled gracefully by tools)
        Chapter.add_chapter(cursor, -1, ["Book", "Prologue"], "Before the story begins")

        negative_chapter = Chapter.read_chapter(cursor, -1)
        assert negative_chapter is not None
        assert negative_chapter.id == -1

        # Test zero chapter index
        Chapter.add_chapter(cursor, 0, ["Book", "Introduction"], "The story begins")

        zero_chapter = Chapter.read_chapter(cursor, 0)
        assert zero_chapter is not None
        assert zero_chapter.id == 0


def test_concurrent_unicode_operations(temp_db: SafeConnection) -> None:
    """Test concurrent operations with unicode content."""
    results = []
    errors = []

    def create_unicode_content(thread_id: int) -> None:
        """Create unicode content in a separate thread."""
        try:
            with temp_db.transaction_cursor() as cursor:
                # Each thread creates content with different unicode scripts
                scripts = ["日本語", "العربية", "हिन्दी", "中文", "Русский"]
                script = scripts[thread_id % len(scripts)]

                conversation = Conversation.create(cursor)
                conversation.add_user_text(f"Thread {thread_id}: {script} content")

                # Small delay to encourage interleaving
                time.sleep(0.001)

                blocks = conversation.blocks
                text_body = blocks[0].text_body
                assert text_body is not None  # Ensure it's not None
                results.append((thread_id, script, text_body))
        except Exception as e:
            errors.append((thread_id, str(e)))

    # Start multiple threads with unicode content
    threads = []
    for i in range(5):
        thread = threading.Thread(target=create_unicode_content, args=(i,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    # Verify no errors and all content preserved
    assert len(errors) == 0, f"Errors occurred: {errors}"
    assert len(results) == 5

    for thread_id, script, text_body in results:
        assert script in text_body
        assert f"Thread {thread_id}" in text_body
