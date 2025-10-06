"""Tests for new Chapter model methods."""

from bookwiki.db import SafeConnection
from bookwiki.models import Block, Chapter, Conversation, WikiPage


def test_get_page_of_chapters_empty(temp_db: SafeConnection) -> None:
    """Test get_page_of_chapters with no chapters."""
    with temp_db.transaction_cursor() as cursor:
        chapters_with_counts = Chapter.get_page_of_chapters(cursor)
        assert chapters_with_counts == []


def test_get_page_of_chapters_no_wiki_pages(temp_db: SafeConnection) -> None:
    """Test get_page_of_chapters with chapters but no wiki pages."""
    with temp_db.transaction_cursor() as cursor:
        # Add some chapters
        Chapter.add_chapter(cursor, 1, ["Book", "Ch1"], "First chapter")
        Chapter.add_chapter(cursor, 2, ["Book", "Ch2"], "Second chapter")
        Chapter.add_chapter(cursor, 3, ["Book", "Ch3"], "Third chapter")

        chapters_with_counts = Chapter.get_page_of_chapters(cursor)

        assert len(chapters_with_counts) == 3
        for _, wiki_count in chapters_with_counts:
            assert wiki_count == 0

        # Verify order
        assert chapters_with_counts[0][0].id == 1
        assert chapters_with_counts[1][0].id == 2
        assert chapters_with_counts[2][0].id == 3


def test_get_page_of_chapters_with_wiki_pages(temp_db: SafeConnection) -> None:
    """Test get_page_of_chapters with wiki pages."""
    with temp_db.transaction_cursor() as cursor:
        # Add chapters
        Chapter.add_chapter(cursor, 1, ["Book", "Ch1"], "First chapter")
        Chapter.add_chapter(cursor, 2, ["Book", "Ch2"], "Second chapter")
        Chapter.add_chapter(cursor, 3, ["Book", "Ch3"], "Third chapter")

        # Create a conversation and block for wiki page creation
        conv = Conversation.create(cursor)
        block = Block.create_text(
            cursor, conv.id, conv.current_generation, "assistant", "test"
        )

        # Add wiki pages to different chapters
        # Chapter 1: 2 distinct slugs
        WikiPage.create(
            cursor,
            1,
            "character-alice",
            block.id,
            "Alice",
            ["Alice"],
            "Summary",
            "Body",
        )
        WikiPage.create(
            cursor, 1, "character-bob", block.id, "Bob", ["Bob"], "Summary", "Body"
        )

        # Chapter 2: 1 slug with multiple versions (should count as 1)
        WikiPage.create(
            cursor,
            2,
            "location-city",
            block.id,
            "City",
            ["City"],
            "Summary v1",
            "Body v1",
        )
        WikiPage.create(
            cursor,
            2,
            "location-city",
            block.id,
            "City",
            ["City"],
            "Summary v2",
            "Body v2",
        )
        WikiPage.create(
            cursor,
            2,
            "location-city",
            block.id,
            "City",
            ["City"],
            "Summary v3",
            "Body v3",
        )

        # Chapter 3: 3 distinct slugs
        WikiPage.create(
            cursor, 3, "item-sword", block.id, "Sword", ["Sword"], "Summary", "Body"
        )
        WikiPage.create(
            cursor, 3, "item-shield", block.id, "Shield", ["Shield"], "Summary", "Body"
        )
        WikiPage.create(
            cursor, 3, "item-potion", block.id, "Potion", ["Potion"], "Summary", "Body"
        )

        chapters_with_counts = Chapter.get_page_of_chapters(cursor)

        assert len(chapters_with_counts) == 3
        assert chapters_with_counts[0][1] == 2  # Chapter 1 has 2 distinct slugs
        assert chapters_with_counts[1][1] == 1  # Chapter 2 has 1 distinct slug
        assert chapters_with_counts[2][1] == 3  # Chapter 3 has 3 distinct slugs


def test_get_page_of_chapters_pagination(temp_db: SafeConnection) -> None:
    """Test get_page_of_chapters with offset and count parameters."""
    with temp_db.transaction_cursor() as cursor:
        # Add 10 chapters
        for i in range(1, 11):
            Chapter.add_chapter(cursor, i, ["Book", f"Ch{i}"], f"Chapter {i}")

        # Get first 5
        page1 = Chapter.get_page_of_chapters(cursor, offset=0, count=5)
        assert len(page1) == 5
        assert [ch.id for ch, _ in page1] == [1, 2, 3, 4, 5]

        # Get next 5
        page2 = Chapter.get_page_of_chapters(cursor, offset=5, count=5)
        assert len(page2) == 5
        assert [ch.id for ch, _ in page2] == [6, 7, 8, 9, 10]

        # Get middle 3
        page3 = Chapter.get_page_of_chapters(cursor, offset=4, count=3)
        assert len(page3) == 3
        assert [ch.id for ch, _ in page3] == [5, 6, 7]

        # Get beyond available chapters
        page4 = Chapter.get_page_of_chapters(cursor, offset=10, count=5)
        assert len(page4) == 0


def test_created_pages_empty(temp_db: SafeConnection) -> None:
    """Test created_pages with no wiki pages."""
    with temp_db.transaction_cursor() as cursor:
        chapter = Chapter.add_chapter(cursor, 1, ["Book", "Ch1"], "First chapter")
        created_pages = chapter.created_pages
        updated_pages = chapter.updated_pages
        assert created_pages == []
        assert updated_pages == []


def test_created_pages_single_chapter(temp_db: SafeConnection) -> None:
    """Test created_pages with pages created in first chapter."""
    with temp_db.transaction_cursor() as cursor:
        chapter = Chapter.add_chapter(cursor, 1, ["Book", "Ch1"], "First chapter")

        # Create a conversation and block for wiki page creation
        conv = Conversation.create(cursor)
        block = Block.create_text(
            cursor, conv.id, conv.current_generation, "assistant", "test"
        )

        # Add single versions of different pages
        wp1 = WikiPage.create(
            cursor,
            1,
            "character-alice",
            block.id,
            "Alice",
            ["Alice"],
            "Alice summary",
            "Alice body",
        )
        wp2 = WikiPage.create(
            cursor,
            1,
            "character-bob",
            block.id,
            "Bob",
            ["Bob"],
            "Bob summary",
            "Bob body",
        )

        created_pages = chapter.created_pages
        updated_pages = chapter.updated_pages

        assert len(created_pages) == 2
        assert len(updated_pages) == 0

        # Pages should be ordered by slug
        assert created_pages[0].slug == "character-alice"
        assert created_pages[0].title == "Alice"
        assert created_pages[0].summary == "Alice summary"
        assert created_pages[0].body == "Alice body"
        assert created_pages[0].id == wp1.id

        assert created_pages[1].slug == "character-bob"
        assert created_pages[1].title == "Bob"
        assert created_pages[1].summary == "Bob summary"
        assert created_pages[1].body == "Bob body"
        assert created_pages[1].id == wp2.id


def test_updated_pages_multiple_chapters(temp_db: SafeConnection) -> None:
    """Test updated_pages with pages updated across chapters."""
    with temp_db.transaction_cursor() as cursor:
        ch1 = Chapter.add_chapter(cursor, 1, ["Book", "Ch1"], "First chapter")
        ch2 = Chapter.add_chapter(cursor, 2, ["Book", "Ch2"], "Second chapter")

        # Create a conversation and block for wiki page creation
        conv = Conversation.create(cursor)
        block = Block.create_text(
            cursor, conv.id, conv.current_generation, "assistant", "test"
        )

        # Chapter 1: Create initial pages
        WikiPage.create(
            cursor,
            1,
            "character-alice",
            block.id,
            "Alice v1",
            ["Alice"],
            "Summary v1",
            "Body v1",
        )
        WikiPage.create(
            cursor,
            1,
            "location-city",
            block.id,
            "City v1",
            ["City"],
            "City summary v1",
            "City body v1",
        )

        # Chapter 2: Update existing pages and create new ones
        wp_alice_latest = WikiPage.create(
            cursor,
            2,
            "character-alice",
            block.id,
            "Alice v2",
            ["Alice"],
            "Summary v2",
            "Body v2",
        )
        wp_bob_new = WikiPage.create(
            cursor,
            2,
            "character-bob",
            block.id,
            "Bob",
            ["Bob"],
            "Bob summary",
            "Bob body",
        )

        # Chapter 1 should have created pages only
        ch1_created = ch1.created_pages
        ch1_updated = ch1.updated_pages
        assert len(ch1_created) == 2  # alice, city
        assert len(ch1_updated) == 0

        # Chapter 2 should have both created and updated pages
        ch2_created = ch2.created_pages
        ch2_updated = ch2.updated_pages
        assert len(ch2_created) == 1  # bob
        assert len(ch2_updated) == 1  # alice

        # Check that the correct pages are returned
        assert ch2_created[0].slug == "character-bob"
        assert ch2_created[0].id == wp_bob_new.id

        assert ch2_updated[0].slug == "character-alice"
        assert ch2_updated[0].title == "Alice v2"
        assert ch2_updated[0].id == wp_alice_latest.id


def test_pages_multiple_versions_same_chapter(temp_db: SafeConnection) -> None:
    """Test that multiple versions in same chapter return latest version only."""
    with temp_db.transaction_cursor() as cursor:
        chapter = Chapter.add_chapter(cursor, 1, ["Book", "Ch1"], "First chapter")

        # Create a conversation and block for wiki page creation
        conv = Conversation.create(cursor)
        block = Block.create_text(
            cursor, conv.id, conv.current_generation, "assistant", "test"
        )

        # Add multiple versions of the same page
        WikiPage.create(
            cursor,
            1,
            "character-alice",
            block.id,
            "Alice v1",
            ["Alice"],
            "Summary v1",
            "Body v1",
        )
        WikiPage.create(
            cursor,
            1,
            "character-alice",
            block.id,
            "Alice v2",
            ["Alice"],
            "Summary v2",
            "Body v2",
        )
        wp_latest = WikiPage.create(
            cursor,
            1,
            "character-alice",
            block.id,
            "Alice v3",
            ["Alice"],
            "Summary v3",
            "Body v3",
        )

        created_pages = chapter.created_pages
        updated_pages = chapter.updated_pages

        # Should return only one page (latest version)
        assert len(created_pages) == 1
        assert len(updated_pages) == 0

        # Should be the latest version
        assert created_pages[0].title == "Alice v3"
        assert created_pages[0].id == wp_latest.id


def test_pages_ordering(temp_db: SafeConnection) -> None:
    """Test that pages are returned ordered by slug."""
    with temp_db.transaction_cursor() as cursor:
        chapter = Chapter.add_chapter(cursor, 1, ["Book", "Ch1"], "First chapter")

        # Create a conversation and block for wiki page creation
        conv = Conversation.create(cursor)
        block = Block.create_text(
            cursor, conv.id, conv.current_generation, "assistant", "test"
        )

        # Add pages in random order
        WikiPage.create(
            cursor, 1, "zebra", block.id, "Zebra", ["Zebra"], "Summary", "Body"
        )
        WikiPage.create(
            cursor, 1, "apple", block.id, "Apple", ["Apple"], "Summary", "Body"
        )
        WikiPage.create(
            cursor, 1, "middle", block.id, "Middle", ["Middle"], "Summary", "Body"
        )
        WikiPage.create(
            cursor, 1, "banana", block.id, "Banana", ["Banana"], "Summary", "Body"
        )

        created_pages = chapter.created_pages
        updated_pages = chapter.updated_pages

        assert len(created_pages) == 4
        assert len(updated_pages) == 0

        # Should be ordered by slug
        created_slugs = [p.slug for p in created_pages]
        assert created_slugs == ["apple", "banana", "middle", "zebra"]


def test_integration_all_methods(temp_db: SafeConnection) -> None:
    """Test all three new methods work together correctly."""
    with temp_db.transaction_cursor() as cursor:
        # Create chapters
        ch1 = Chapter.add_chapter(cursor, 1, ["Book", "Ch1"], "First chapter")
        ch2 = Chapter.add_chapter(cursor, 2, ["Book", "Ch2"], "Second chapter")

        # Create a conversation and block for wiki page creation
        conv = Conversation.create(cursor)
        block = Block.create_text(
            cursor, conv.id, conv.current_generation, "assistant", "test"
        )

        # Add pages with multiple versions to chapter 1
        WikiPage.create(
            cursor, 1, "alice", block.id, "Alice v1", ["Alice"], "Summary v1", "Body v1"
        )
        WikiPage.create(
            cursor, 1, "alice", block.id, "Alice v2", ["Alice"], "Summary v2", "Body v2"
        )
        WikiPage.create(
            cursor, 1, "bob", block.id, "Bob", ["Bob"], "Bob summary", "Bob body"
        )

        # Add pages to chapter 2
        WikiPage.create(
            cursor,
            2,
            "charlie",
            block.id,
            "Charlie",
            ["Charlie"],
            "Charlie summary",
            "Charlie body",
        )

        # Test get_page_of_chapters
        chapters_with_counts = Chapter.get_page_of_chapters(cursor)
        assert len(chapters_with_counts) == 2
        assert chapters_with_counts[0][0].id == 1
        assert chapters_with_counts[0][1] == 2  # 2 distinct slugs in ch1
        assert chapters_with_counts[1][0].id == 2
        assert chapters_with_counts[1][1] == 1  # 1 distinct slug in ch2

        # Test created_pages and updated_pages
        ch1_created = ch1.created_pages
        ch1_updated = ch1.updated_pages
        assert len(ch1_created) == 2  # alice, bob
        assert len(ch1_updated) == 0
        assert ch1_created[0].slug == "alice"
        assert ch1_created[0].title == "Alice v2"  # Latest version
        assert ch1_created[1].slug == "bob"

        ch2_created = ch2.created_pages
        ch2_updated = ch2.updated_pages
        assert len(ch2_created) == 1  # charlie
        assert len(ch2_updated) == 0
        assert ch2_created[0].slug == "charlie"


def test_wiki_stats_empty_chapter(temp_db: SafeConnection) -> None:
    """Test wiki_stats with no wiki pages."""
    with temp_db.transaction_cursor() as cursor:
        chapter = Chapter.add_chapter(cursor, 1, ["Book", "Ch1"], "First chapter")

        assert len(chapter.created_pages) == 0
        assert len(chapter.updated_pages) == 0


def test_wiki_stats_all_new_pages(temp_db: SafeConnection) -> None:
    """Test wiki_stats when all pages are newly created."""
    with temp_db.transaction_cursor() as cursor:
        chapter = Chapter.add_chapter(cursor, 1, ["Book", "Ch1"], "First chapter")

        # Create conversation and block for wiki page creation
        conv = Conversation.create(cursor)
        block = Block.create_text(
            cursor, conv.id, conv.current_generation, "assistant", "test"
        )

        # Create 3 new wiki pages in this chapter
        WikiPage.create(
            cursor, 1, "alice", block.id, "Alice", ["Alice"], "Summary", "Body"
        )
        WikiPage.create(cursor, 1, "bob", block.id, "Bob", ["Bob"], "Summary", "Body")
        WikiPage.create(
            cursor, 1, "charlie", block.id, "Charlie", ["Charlie"], "Summary", "Body"
        )

        assert len(chapter.created_pages) == 3
        assert len(chapter.updated_pages) == 0


def test_wiki_stats_all_updated_pages(temp_db: SafeConnection) -> None:
    """Test wiki_stats when all pages are updates to existing ones."""
    with temp_db.transaction_cursor() as cursor:
        # Create chapters
        Chapter.add_chapter(cursor, 1, ["Book", "Ch1"], "First chapter")
        chapter2 = Chapter.add_chapter(cursor, 2, ["Book", "Ch2"], "Second chapter")

        # Create conversation and block
        conv = Conversation.create(cursor)
        block = Block.create_text(
            cursor, conv.id, conv.current_generation, "assistant", "test"
        )

        # Create original pages in chapter 1
        WikiPage.create(
            cursor, 1, "alice", block.id, "Alice v1", ["Alice"], "Summary v1", "Body v1"
        )
        WikiPage.create(
            cursor, 1, "bob", block.id, "Bob v1", ["Bob"], "Summary v1", "Body v1"
        )

        # Update existing pages in chapter 2
        WikiPage.create(
            cursor, 2, "alice", block.id, "Alice v2", ["Alice"], "Summary v2", "Body v2"
        )
        WikiPage.create(
            cursor, 2, "bob", block.id, "Bob v2", ["Bob"], "Summary v2", "Body v2"
        )

        assert len(chapter2.created_pages) == 0
        assert len(chapter2.updated_pages) == 2


def test_wiki_stats_mixed_pages(temp_db: SafeConnection) -> None:
    """Test wiki_stats with a mix of new and updated pages."""
    with temp_db.transaction_cursor() as cursor:
        # Create chapters
        Chapter.add_chapter(cursor, 1, ["Book", "Ch1"], "First chapter")
        chapter2 = Chapter.add_chapter(cursor, 2, ["Book", "Ch2"], "Second chapter")

        # Create conversation and block
        conv = Conversation.create(cursor)
        block = Block.create_text(
            cursor, conv.id, conv.current_generation, "assistant", "test"
        )

        # Create original pages in chapter 1
        WikiPage.create(
            cursor, 1, "alice", block.id, "Alice v1", ["Alice"], "Summary v1", "Body v1"
        )
        WikiPage.create(
            cursor, 1, "bob", block.id, "Bob v1", ["Bob"], "Summary v1", "Body v1"
        )

        # In chapter 2: update existing pages and create new ones
        WikiPage.create(
            cursor,
            2,
            "alice",
            block.id,
            "Alice v2",
            ["Alice"],
            "Summary v2",
            "Body v2",  # Updated
        )
        WikiPage.create(
            cursor,
            2,
            "charlie",
            block.id,
            "Charlie",
            ["Charlie"],
            "Summary",
            "Body",  # New
        )
        WikiPage.create(
            cursor,
            2,
            "diana",
            block.id,
            "Diana",
            ["Diana"],
            "Summary",
            "Body",  # New
        )

        assert len(chapter2.created_pages) == 2  # charlie, diana
        assert len(chapter2.updated_pages) == 1  # alice


def test_wiki_stats_multiple_versions_same_chapter(temp_db: SafeConnection) -> None:
    """Test wiki_stats when same slug has multiple versions in the same chapter."""
    with temp_db.transaction_cursor() as cursor:
        chapter = Chapter.add_chapter(cursor, 1, ["Book", "Ch1"], "First chapter")

        # Create conversation and block
        conv = Conversation.create(cursor)
        block = Block.create_text(
            cursor, conv.id, conv.current_generation, "assistant", "test"
        )

        # Create multiple versions of the same page in the same chapter
        WikiPage.create(
            cursor, 1, "alice", block.id, "Alice v1", ["Alice"], "Summary v1", "Body v1"
        )
        WikiPage.create(
            cursor, 1, "alice", block.id, "Alice v2", ["Alice"], "Summary v2", "Body v2"
        )
        WikiPage.create(
            cursor, 1, "alice", block.id, "Alice v3", ["Alice"], "Summary v3", "Body v3"
        )

        # Create another page with multiple versions
        WikiPage.create(
            cursor, 1, "bob", block.id, "Bob v1", ["Bob"], "Summary v1", "Body v1"
        )
        WikiPage.create(
            cursor, 1, "bob", block.id, "Bob v2", ["Bob"], "Summary v2", "Body v2"
        )

        # Should count unique slugs, not individual wiki page records
        assert len(chapter.created_pages) == 2  # alice, bob
        assert len(chapter.updated_pages) == 0


def test_wiki_stats_complex_scenario(temp_db: SafeConnection) -> None:
    """Test wiki_stats in a complex multi-chapter scenario."""
    with temp_db.transaction_cursor() as cursor:
        # Create chapters
        Chapter.add_chapter(cursor, 1, ["Book", "Ch1"], "First chapter")
        Chapter.add_chapter(cursor, 2, ["Book", "Ch2"], "Second chapter")
        chapter3 = Chapter.add_chapter(cursor, 3, ["Book", "Ch3"], "Third chapter")

        # Create conversation and block
        conv = Conversation.create(cursor)
        block = Block.create_text(
            cursor, conv.id, conv.current_generation, "assistant", "test"
        )

        # Chapter 1: Create initial pages
        WikiPage.create(
            cursor, 1, "alice", block.id, "Alice v1", ["Alice"], "Summary v1", "Body v1"
        )
        WikiPage.create(
            cursor, 1, "bob", block.id, "Bob v1", ["Bob"], "Summary v1", "Body v1"
        )

        # Chapter 2: Update alice, create charlie
        WikiPage.create(
            cursor, 2, "alice", block.id, "Alice v2", ["Alice"], "Summary v2", "Body v2"
        )
        WikiPage.create(
            cursor,
            2,
            "charlie",
            block.id,
            "Charlie v1",
            ["Charlie"],
            "Summary v1",
            "Body v1",
        )

        # Chapter 3: Update alice and charlie, create diana and eve
        WikiPage.create(
            cursor,
            3,
            "alice",
            block.id,
            "Alice v3",
            ["Alice"],
            "Summary v3",
            "Body v3",  # Updated
        )
        WikiPage.create(
            cursor,
            3,
            "charlie",
            block.id,
            "Charlie v2",
            ["Charlie"],
            "Summary v2",
            "Body v2",  # Updated
        )
        WikiPage.create(
            cursor,
            3,
            "diana",
            block.id,
            "Diana",
            ["Diana"],
            "Summary",
            "Body",  # New
        )
        WikiPage.create(
            cursor,
            3,
            "eve",
            block.id,
            "Eve",
            ["Eve"],
            "Summary",
            "Body",  # New
        )

        assert len(chapter3.created_pages) == 2  # diana, eve
        assert len(chapter3.updated_pages) == 2  # alice, charlie


def test_wiki_stats_skip_chapters(temp_db: SafeConnection) -> None:
    """Test wiki_stats when there are gaps in chapter numbers."""
    with temp_db.transaction_cursor() as cursor:
        # Create non-consecutive chapters
        Chapter.add_chapter(cursor, 1, ["Book", "Ch1"], "First chapter")
        Chapter.add_chapter(cursor, 5, ["Book", "Ch5"], "Fifth chapter")
        chapter10 = Chapter.add_chapter(cursor, 10, ["Book", "Ch10"], "Tenth chapter")

        # Create conversation and block
        conv = Conversation.create(cursor)
        block = Block.create_text(
            cursor, conv.id, conv.current_generation, "assistant", "test"
        )

        # Chapter 1: Create pages
        WikiPage.create(
            cursor, 1, "alice", block.id, "Alice v1", ["Alice"], "Summary v1", "Body v1"
        )

        # Chapter 5: Update alice, create bob
        WikiPage.create(
            cursor, 5, "alice", block.id, "Alice v2", ["Alice"], "Summary v2", "Body v2"
        )
        WikiPage.create(cursor, 5, "bob", block.id, "Bob", ["Bob"], "Summary", "Body")

        # Chapter 10: Update alice, create charlie
        WikiPage.create(
            cursor,
            10,
            "alice",
            block.id,
            "Alice v3",
            ["Alice"],
            "Summary v3",
            "Body v3",  # Updated
        )
        WikiPage.create(
            cursor,
            10,
            "charlie",
            block.id,
            "Charlie",
            ["Charlie"],
            "Summary",
            "Body",  # New
        )

        assert len(chapter10.created_pages) == 1  # charlie
        assert len(chapter10.updated_pages) == 1  # alice (existed in chapters 1 and 5)


def test_wiki_stats_cached_property(temp_db: SafeConnection) -> None:
    """Test that created_pages and updated_pages are properly cached."""
    with temp_db.transaction_cursor() as cursor:
        chapter = Chapter.add_chapter(cursor, 1, ["Book", "Ch1"], "First chapter")

        # First access should calculate and cache
        created1 = chapter.created_pages
        updated1 = chapter.updated_pages
        assert len(created1) == 0
        assert len(updated1) == 0

        # Second access should return the same cached objects
        created2 = chapter.created_pages
        updated2 = chapter.updated_pages
        assert created1 is created2
        assert updated1 is updated2

        # Create new chapter instance - should not have cached value
        chapter2 = Chapter.read_chapter(cursor, 1)
        assert chapter2 is not None

        # This will calculate fresh (but same result)
        created3 = chapter2.created_pages
        updated3 = chapter2.updated_pages
        assert len(created3) == 0
        assert len(updated3) == 0
        assert created3 is not created1  # Different object, same values
        assert updated3 is not updated1  # Different object, same values


def test_wiki_pages_immutable(temp_db: SafeConnection) -> None:
    """Test that created_pages and updated_pages return lists."""
    with temp_db.transaction_cursor() as cursor:
        chapter = Chapter.add_chapter(cursor, 1, ["Book", "Ch1"], "First chapter")

        # Get the cached properties
        created = chapter.created_pages
        updated = chapter.updated_pages

        # They should be lists
        assert isinstance(created, list)
        assert isinstance(updated, list)
        assert len(created) == 0
        assert len(updated) == 0
