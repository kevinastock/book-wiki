from bookwiki.db import SafeConnection
from bookwiki.models import Block, Chapter, Conversation, WikiPage
from bookwiki.models.wikipage import _convert_values_to_ranks


def test_convert_values_to_ranks() -> None:
    """Test the _convert_values_to_ranks helper function."""
    # Test basic ranking
    assert _convert_values_to_ranks([3, 1, 2]) == [1, 3, 2]

    # Test ties - both 3's get rank 1, next gets rank 3 (not rank 2)
    assert _convert_values_to_ranks([3, 3, 2]) == [1, 1, 3]

    # Test all same values
    assert _convert_values_to_ranks([5, 5, 5]) == [1, 1, 1]

    # Test empty list
    assert _convert_values_to_ranks([]) == []

    # Test single value
    assert _convert_values_to_ranks([42]) == [1]

    # Test complex ties: [5, 3, 5, 2, 3] -> [1, 3, 1, 5, 3]
    assert _convert_values_to_ranks([5, 3, 5, 2, 3]) == [1, 3, 1, 5, 3]


def test_get_all_pages_chapter_rrf_ordering(temp_db: SafeConnection) -> None:
    """Test that get_all_pages_chapter orders pages correctly using RRF."""
    with temp_db.transaction_cursor() as cursor:
        # Create test chapters
        ch1 = Chapter.add_chapter(cursor, 1, ["Chapter 1"], "Content 1")
        ch2 = Chapter.add_chapter(cursor, 2, ["Chapter 2"], "Content 2")
        ch3 = Chapter.add_chapter(cursor, 3, ["Chapter 3"], "Content 3")

        # Create a test conversation and block for wiki page creation
        conversation = Conversation.create(cursor)
        block = Block.create_text(
            cursor,
            conversation.id,
            conversation.current_generation,
            "user",
            "test content",
        )

        # Create test wiki pages with different characteristics:
        # Create wiki pages in realistic chapter order with proper current page copying

        # Chapter 1: Create Page A and Page D
        WikiPage.create(
            cursor,
            ch1.id,
            "page-a",
            block.id,
            "Page A",
            ["Page A"],
            "Summary A",
            "Short body",
        )
        WikiPage.create(
            cursor,
            ch1.id,
            "page-d",
            block.id,
            "Page D",
            ["Page D"],
            "Summary D",
            "Short",
        )

        # Chapter 2: Copy current pages from ch1, then create/update pages
        WikiPage.copy_current_for_new_chapter(cursor, ch2.id)
        WikiPage.create(
            cursor,
            ch2.id,
            "page-a",
            block.id,
            "Page A v2",
            ["Page A"],
            "Summary A",
            "Medium length body",
        )  # Update Page A
        WikiPage.create(
            cursor,
            ch2.id,
            "page-b",
            block.id,
            "Page B",
            ["Page B"],
            "Summary B",
            "Medium body",
        )  # New Page B

        # Chapter 3: Copy current pages from ch2, then create/update pages
        WikiPage.copy_current_for_new_chapter(cursor, ch3.id)
        WikiPage.create(
            cursor,
            ch3.id,
            "page-a",
            block.id,
            "Page A v3",
            ["Page A"],
            "Summary A",
            "Very long body content here",
        )  # Update Page A
        WikiPage.create(
            cursor,
            ch3.id,
            "page-b",
            block.id,
            "Page B v2",
            ["Page B"],
            "Summary B",
            "Long body content",
        )  # Update Page B
        WikiPage.create(
            cursor,
            ch3.id,
            "page-c",
            block.id,
            "Page C",
            ["Page C"],
            "Summary C",
            "Longest body content of all pages",
        )  # New Page C

        # Get all pages at chapter 3
        pages = WikiPage.get_all_pages_chapter(cursor, ch3.id)

        # Verify we get the expected pages
        assert len(pages) == 4
        page_slugs = [p.slug for p in pages]
        assert set(page_slugs) == {"page-a", "page-b", "page-c", "page-d"}

        # Expected ranking logic:
        # 1. Chapter ranking (DESC): All A, B, C have chapter_id=3, D has chapter_id=1
        #    So: [A, B, C] tie for first, D last
        # 2. Distinct chapters ranking (DESC): A=3, B=2, C=1, D=1
        #    So: A first, B second, [C, D] tie for third
        #
        # RRF will combine these rankings:
        # - A gets good scores from both rankings (1st in both or nearly)
        # - B gets medium scores (1st in chapter, 2nd in distinct)
        # - C gets mixed scores (1st in chapter, tied 3rd in distinct)
        # - D gets poor scores (4th in chapter, tied 3rd in distinct)
        #
        # Expected order: A should be first (best RRF score)
        # For ties, longer body length wins

        # Page A should be first (appears in most chapters and is recent)
        assert pages[0].slug == "page-a"

        # Page C should rank highly due to longest body breaking ties
        # but Page B might rank higher due to appearing in more chapters
        page_positions = {p.slug: i for i, p in enumerate(pages)}

        # A should be first due to high distinct_chapters + recent chapter
        assert page_positions["page-a"] == 0

        # D should be last due to old chapter_id and low distinct_chapters
        assert page_positions["page-d"] == 3

        # B should rank higher than C despite C having longer body,
        # because B appears in more chapters (distinct_chapters: 2 vs 1)
        assert page_positions["page-b"] < page_positions["page-c"]


def test_get_all_pages_chapter_empty(temp_db: SafeConnection) -> None:
    """Test that empty result is handled correctly."""
    with temp_db.transaction_cursor() as cursor:
        ch = Chapter.add_chapter(cursor, 1, ["Chapter 1"], "Content")
        pages = WikiPage.get_all_pages_chapter(cursor, ch.id)
        assert pages == []


def test_get_all_pages_chapter_single_page(temp_db: SafeConnection) -> None:
    """Test with single page."""
    with temp_db.transaction_cursor() as cursor:
        ch = Chapter.add_chapter(cursor, 1, ["Chapter 1"], "Content")
        conversation = Conversation.create(cursor)
        block = Block.create_text(
            cursor, conversation.id, conversation.current_generation, "user", "test"
        )

        WikiPage.create(
            cursor, ch.id, "test-page", block.id, "Test", ["Test"], "Summary", "Body"
        )

        pages = WikiPage.get_all_pages_chapter(cursor, ch.id)
        assert len(pages) == 1
        assert pages[0].slug == "test-page"
