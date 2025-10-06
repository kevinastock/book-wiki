"""Tests for bookwiki wiki tools."""

import numpy as np
import pytest

from bookwiki.db import SafeConnection
from bookwiki.models import Block, Chapter, Conversation, WikiPage
from bookwiki.search import (
    _compute_similarity_scores,
    _convert_name_scores_to_slug_scores,
    _rank_slugs_by_query,
    _reciprocal_rank_fusion,
)
from bookwiki.tools.base import LLMSolvableError
from bookwiki.tools.wiki import (
    ReadWikiPage,
    SearchWikiByName,
    WriteWikiPage,
)


def test_read_wiki_page_success(temp_db: SafeConnection) -> None:
    """Test successfully reading an existing wiki page."""
    with temp_db.transaction_cursor() as cursor:
        # Set up test data
        chapter = Chapter.add_chapter(
            cursor, 1, ["Book 1", "Chapter 1"], "Chapter content..."
        )
        conversation = Conversation.create(cursor)
        chapter.set_conversation_id(conversation)

        # Create a wiki page first
        create_block = Block.create_text(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            role="assistant",
            text="Creating page",
        )

        create_block.write_wiki_page(
            chapter_id=1,
            slug="test-character",
            title="Test Character",
            names=["Test Character", "The Tester"],
            summary="A character for testing",
            body="This is the full wiki page body with detailed information.",
        )

        # Now try to read it
        read_block = Block.create_tool_use(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            name="ReadWikiPage",
            use_id="read_1",
            params='{"slug": "test-character"}',
        )

        tool = ReadWikiPage(
            tool_id="read_1",
            tool_name="ReadWikiPage",
            slug="test-character",
        )

        tool.apply(read_block)

        # Check the response using model method
        updated_block = Block.get_by_id(cursor, read_block.id)
        assert updated_block is not None
        assert updated_block.tool_response is not None
        # Note: The formatting in the actual tool has a bug - it doesn't use f-strings
        assert "# Test Character" in updated_block.tool_response
        assert (
            "Known names: ['Test Character', 'The Tester']"
            in updated_block.tool_response
        )
        assert updated_block.errored is False


def test_read_wiki_page_nonexistent(temp_db: SafeConnection) -> None:
    """Test reading a wiki page that doesn't exist."""
    with temp_db.transaction_cursor() as cursor:
        chapter = Chapter.add_chapter(
            cursor, 1, ["Book 1", "Chapter 1"], "Chapter content..."
        )
        conversation = Conversation.create(cursor)
        chapter.set_conversation_id(conversation)

        block = Block.create_tool_use(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            name="ReadWikiPage",
            use_id="read_missing",
            params='{"slug": "nonexistent"}',
        )

        tool = ReadWikiPage(
            tool_id="read_missing",
            tool_name="ReadWikiPage",
            slug="nonexistent",
        )

        tool.apply(block)

        # Should get an error response
        updated_block = Block.get_by_id(cursor, block.id)
        assert updated_block is not None
        assert updated_block.tool_response is not None
        assert updated_block.tool_response == "No page exists with slug 'nonexistent'"
        assert updated_block.errored is True


def test_write_wiki_page_create_new(temp_db: SafeConnection) -> None:
    """Test creating a new wiki page."""
    with temp_db.transaction_cursor() as cursor:
        chapter = Chapter.add_chapter(
            cursor, 1, ["Book 1", "Chapter 1"], "Chapter content..."
        )
        conversation = Conversation.create(cursor)
        chapter.set_conversation_id(conversation)

        block = Block.create_tool_use(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            name="WriteWikiPage",
            use_id="write_new",
            params=(
                '{"slug": "new-character", '
                '"title": "New Character", "names": ["New Character", "Newcomer"], '
                '"summary": "A brand new character", "body": "Full description...", '
                '"create": true}'
            ),
        )

        tool = WriteWikiPage(
            tool_id="write_new",
            tool_name="WriteWikiPage",
            slug="new-character",
            title="New Character",
            names=["New Character", "Newcomer"],
            summary="A brand new character",
            body="Full description...",
            create=True,
        )

        tool.apply(block)

        # Check success response using model method
        updated_block = Block.get_by_id(cursor, block.id)
        assert updated_block is not None
        assert updated_block.tool_response == "Wrote wiki page"
        assert updated_block.errored is False

        # Verify the page was actually created
        created_page = WikiPage.read_page_at(cursor, "new-character", 1)
        assert created_page is not None
        assert created_page.title == "New Character"
        assert set(created_page.names) == {"New Character", "Newcomer"}
        assert created_page.summary == "A brand new character"
        assert created_page.body == "Full description..."


def test_write_wiki_page_create_already_exists(temp_db: SafeConnection) -> None:
    """Test creating a wiki page when it already exists."""
    with temp_db.transaction_cursor() as cursor:
        chapter = Chapter.add_chapter(
            cursor, 1, ["Book 1", "Chapter 1"], "Chapter content..."
        )
        conversation = Conversation.create(cursor)
        chapter.set_conversation_id(conversation)

        # Create an existing page
        existing_block = Block.create_text(
            cursor,
            conversation.id,
            conversation.current_generation,
            "assistant",
            "Creating",
        )
        existing_block.write_wiki_page(
            chapter_id=1,
            slug="existing",
            title="Existing Page",
            names=["Existing"],
            summary="Already here",
            body="Existing content",
        )

        # Try to create with same slug
        block = Block.create_tool_use(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            name="WriteWikiPage",
            use_id="write_duplicate",
            params=(
                '{"slug": "existing", '
                '"title": "New Title", "names": ["New Name"], '
                '"summary": "New summary", "body": "New body", '
                '"create": true}'
            ),
        )

        tool = WriteWikiPage(
            tool_id="write_duplicate",
            tool_name="WriteWikiPage",
            slug="existing",
            title="New Title",
            names=["New Name"],
            summary="New summary",
            body="New body",
            create=True,
        )

        tool.apply(block)

        # Should get an error using model method
        updated_block = Block.get_by_id(cursor, block.id)
        assert updated_block is not None
        assert (
            updated_block.tool_response
            == "That slug already exists, but create was specified"
        )
        assert updated_block.errored is True


def test_write_wiki_page_create_missing_fields(temp_db: SafeConnection) -> None:
    """Test creating a wiki page with missing required fields."""
    with temp_db.transaction_cursor() as cursor:
        chapter = Chapter.add_chapter(
            cursor, 1, ["Book 1", "Chapter 1"], "Chapter content..."
        )
        conversation = Conversation.create(cursor)
        chapter.set_conversation_id(conversation)

        block = Block.create_tool_use(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            name="WriteWikiPage",
            use_id="write_incomplete",
            params=('{"slug": "incomplete", "title": "Title Only", "create": true}'),
        )

        tool = WriteWikiPage(
            tool_id="write_incomplete",
            tool_name="WriteWikiPage",
            slug="incomplete",
            title="Title Only",
            names=None,  # Missing
            summary=None,  # Missing
            body=None,  # Missing
            create=True,
        )

        tool.apply(block)

        # Should get an error using model method
        updated_block = Block.get_by_id(cursor, block.id)
        assert updated_block is not None
        assert (
            updated_block.tool_response
            == "All fields must be set when creating a new page"
        )
        assert updated_block.errored is True


def test_write_wiki_page_update_existing(temp_db: SafeConnection) -> None:
    """Test updating an existing wiki page."""
    with temp_db.transaction_cursor() as cursor:
        chapter = Chapter.add_chapter(
            cursor, 1, ["Book 1", "Chapter 1"], "Chapter content..."
        )
        conversation = Conversation.create(cursor)
        chapter.set_conversation_id(conversation)

        # Create an initial page
        create_block = Block.create_text(
            cursor,
            conversation.id,
            conversation.current_generation,
            "assistant",
            "Creating",
        )
        create_block.write_wiki_page(
            chapter_id=1,
            slug="updatable",
            title="Original Title",
            names=["Original Name"],
            summary="Original summary",
            body="Original body",
        )

        # Update it with partial fields
        update_block = Block.create_tool_use(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            name="WriteWikiPage",
            use_id="write_update",
            params=(
                '{"slug": "updatable", "summary": "Updated summary", "create": false}'
            ),
        )

        tool = WriteWikiPage(
            tool_id="write_update",
            tool_name="WriteWikiPage",
            slug="updatable",
            title=None,  # Keep original
            names=None,  # Keep original
            summary="Updated summary",  # Update this
            body=None,  # Keep original
            create=False,
        )

        tool.apply(update_block)

        # Check success using model method
        updated_block = Block.get_by_id(cursor, update_block.id)
        assert updated_block is not None
        assert updated_block.tool_response == "Wrote wiki page"
        assert updated_block.errored is False

        # Verify the update
        updated_page = WikiPage.read_page_at(cursor, "updatable", 1)
        assert updated_page is not None
        assert updated_page.title == "Original Title"  # Unchanged
        assert updated_page.names == ["Original Name"]  # Unchanged
        assert updated_page.summary == "Updated summary"  # Changed
        assert updated_page.body == "Original body"  # Unchanged


def test_write_wiki_page_update_nonexistent(temp_db: SafeConnection) -> None:
    """Test updating a wiki page that doesn't exist."""
    with temp_db.transaction_cursor() as cursor:
        chapter = Chapter.add_chapter(
            cursor, 1, ["Book 1", "Chapter 1"], "Chapter content..."
        )
        conversation = Conversation.create(cursor)
        chapter.set_conversation_id(conversation)

        block = Block.create_tool_use(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            name="WriteWikiPage",
            use_id="write_update_missing",
            params=(
                '{"slug": "nonexistent", "summary": "New summary", "create": false}'
            ),
        )

        tool = WriteWikiPage(
            tool_id="write_update_missing",
            tool_name="WriteWikiPage",
            slug="nonexistent",
            title=None,
            names=None,
            summary="New summary",
            body=None,
            create=False,
        )

        tool.apply(block)

        # Should get an error using model method
        updated_block = Block.get_by_id(cursor, block.id)
        assert updated_block is not None
        assert (
            updated_block.tool_response
            == "No such slug exists, but create was not specified"
        )
        assert updated_block.errored is True


def test_search_wiki_by_name_fuzzy_matching(temp_db: SafeConnection) -> None:
    """Test searching wiki pages with fuzzy name matching."""
    with temp_db.transaction_cursor() as cursor:
        # Set up test data
        chapter = Chapter.add_chapter(
            cursor, 1, ["Book 1", "Chapter 1"], "Chapter content..."
        )
        conversation = Conversation.create(cursor)
        chapter.set_conversation_id(conversation)

        # Create several wiki pages with various names
        blocks = []
        for i in range(3):
            block = Block.create_text(
                cursor,
                conversation.id,
                conversation.current_generation,
                "assistant",
                f"Creating page {i}",
            )
            blocks.append(block)

        blocks[0].write_wiki_page(
            chapter_id=1,
            slug="rand-althor",
            title="Rand al'Thor",
            names=["Rand al'Thor", "Dragon Reborn", "Lord Dragon"],
            summary="The Dragon Reborn",
            body="The prophesied savior...",
        )

        blocks[1].write_wiki_page(
            chapter_id=1,
            slug="perrin-aybara",
            title="Perrin Aybara",
            names=["Perrin Aybara", "Young Bull", "Goldeneyes"],
            summary="The Wolf Brother",
            body="A blacksmith's apprentice...",
        )

        blocks[2].write_wiki_page(
            chapter_id=1,
            slug="mat-cauthon",
            title="Mat Cauthon",
            names=["Mat Cauthon", "Matrim Cauthon", "Son of Battles"],
            summary="The Gambler",
            body="A farmer's son with incredible luck...",
        )

        # Create a search block with typo/fuzzy query
        search_block = Block.create_tool_use(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            name="SearchWikiByName",
            use_id="search_1",
            params='{"names": ["Rand", "Perin", "Matt"]}',  # Note the typos
        )

        tool = SearchWikiByName(
            tool_id="search_1",
            tool_name="SearchWikiByName",
            names=[
                "Rand",
                "Perin",
                "Matt",
            ],  # Typos: Perin instead of Perrin, Matt instead of Mat
            results_page=1,
        )

        # Run the search and verify it works correctly
        tool.apply(search_block)

        # Check that we got a response using model method
        updated_search_block = Block.get_by_id(cursor, search_block.id)
        assert updated_search_block is not None
        assert updated_search_block.tool_response is not None
        assert updated_search_block.errored is False

        # Should contain search results with the fuzzy matched characters
        response = updated_search_block.tool_response
        assert "Search Results" in response
        # Should find matches for the typos: "Rand" -> "Rand al'Thor",
        # "Perin" -> "Perrin Aybara", "Matt" -> "Mat Cauthon"
        assert (
            "rand-althor" in response
            or "perrin-aybara" in response
            or "mat-cauthon" in response
        )


def test_search_wiki_by_name_empty_database(temp_db: SafeConnection) -> None:
    """Test searching when no wiki pages exist."""
    with temp_db.transaction_cursor() as cursor:
        # Set up chapter but no wiki pages
        chapter = Chapter.add_chapter(
            cursor, 1, ["Book 1", "Chapter 1"], "Chapter content..."
        )
        conversation = Conversation.create(cursor)
        chapter.set_conversation_id(conversation)

        search_block = Block.create_tool_use(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            name="SearchWikiByName",
            use_id="search_empty",
            params='{"names": ["Any Name"]}',
        )

        tool = SearchWikiByName(
            tool_id="search_empty",
            tool_name="SearchWikiByName",
            names=["Any Name"],
            results_page=1,
        )

        tool.apply(search_block)

        # Should get a response saying no pages found using model method
        updated_search_block = Block.get_by_id(cursor, search_block.id)
        assert updated_search_block is not None
        assert updated_search_block.tool_response == "No wiki pages found."
        assert updated_search_block.errored is False


def test_search_wiki_caching(temp_db: SafeConnection) -> None:
    """Test that similarity score computation is cached."""
    with temp_db.transaction_cursor() as cursor:
        # Set up test data
        chapter = Chapter.add_chapter(
            cursor, 1, ["Book 1", "Chapter 1"], "Chapter content..."
        )
        conversation = Conversation.create(cursor)
        chapter.set_conversation_id(conversation)

        # Create a wiki page
        block = Block.create_text(
            cursor,
            conversation.id,
            conversation.current_generation,
            "assistant",
            "Creating page",
        )
        block.write_wiki_page(
            chapter_id=1,
            slug="test-character",
            title="Test Character",
            names=["Test Character", "The Tester", "Testing Person"],
            summary="A test character",
            body="For testing purposes",
        )

        # Import the cached function to check cache info
        # Clear the cache before testing
        _compute_similarity_scores.cache_clear()

        # Create two search blocks with identical queries
        search_block1 = Block.create_tool_use(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            name="SearchWikiByName",
            use_id="search_1",
            params='{"names": ["Test", "Tester"], "results_page": 1}',
        )

        search_block2 = Block.create_tool_use(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            name="SearchWikiByName",
            use_id="search_2",
            params='{"names": ["Test", "Tester"], "results_page": 2}',
        )

        from bookwiki.tools.wiki import SearchWikiByName

        # First search - should compute scores
        tool1 = SearchWikiByName(
            tool_id="search_1",
            tool_name="SearchWikiByName",
            names=["Test", "Tester"],
            results_page=1,
        )

        # Check initial cache state
        initial_cache_info = _compute_similarity_scores.cache_info()

        # First search - should compute scores and cache them
        tool1.apply(search_block1)

        # Check cache after first call
        after_first = _compute_similarity_scores.cache_info()
        assert after_first.misses == initial_cache_info.misses + 1  # One cache miss
        assert after_first.hits == initial_cache_info.hits  # No cache hits yet

        # Verify first search worked using model method
        updated_search_block1 = Block.get_by_id(cursor, search_block1.id)
        assert updated_search_block1 is not None
        assert updated_search_block1.tool_response is not None
        assert updated_search_block1.errored is False

        # Second search with same queries - should use cached scores
        tool2 = SearchWikiByName(
            tool_id="search_2",
            tool_name="SearchWikiByName",
            names=["Test", "Tester"],
            results_page=2,  # Different page but same queries
        )

        tool2.apply(search_block2)

        # Check cache after second call
        after_second = _compute_similarity_scores.cache_info()
        assert after_second.hits == after_first.hits + 1  # One cache hit
        assert after_second.misses == after_first.misses  # No additional misses

        # Verify second search worked using model method
        updated_search_block2 = Block.get_by_id(cursor, search_block2.id)
        assert updated_search_block2 is not None
        assert updated_search_block2.tool_response is not None
        assert updated_search_block2.errored is False


def test_convert_name_scores_to_slug_scores() -> None:
    """Test the _convert_name_scores_to_slug_scores function directly."""
    # Set up test data
    # Three names, two slugs (first two names map to slug1, third to slug2)
    choices = ["Name A", "Name B", "Name C"]
    name_to_slugs = {
        "Name A": ["slug1"],
        "Name B": ["slug1"],  # Both A and B map to slug1
        "Name C": ["slug2"],
    }

    # Create a score matrix for 2 queries and 3 names
    # Query 0: scores [10, 20, 30] for names A, B, C
    # Query 1: scores [40, 35, 25] for names A, B, C
    name_scores = np.array(
        [
            [10, 20, 30],  # Query 0 scores
            [40, 35, 25],  # Query 1 scores
        ]
    )

    # Convert to slug scores
    slug_scores, slugs = _convert_name_scores_to_slug_scores(
        name_scores, choices, name_to_slugs
    )

    # Check the results
    assert slugs == ["slug1", "slug2"]  # Sorted order
    assert slug_scores.shape == (2, 2)  # 2 queries, 2 slugs

    # For Query 0:
    # - slug1 should get max(10, 20) = 20 (max of Name A and Name B scores)
    # - slug2 should get 30 (Name C score)
    assert slug_scores[0, 0] == 20  # Query 0, slug1
    assert slug_scores[0, 1] == 30  # Query 0, slug2

    # For Query 1:
    # - slug1 should get max(40, 35) = 40
    # - slug2 should get 25
    assert slug_scores[1, 0] == 40  # Query 1, slug1
    assert slug_scores[1, 1] == 25  # Query 1, slug2


def test_convert_name_scores_to_slug_scores_overlapping() -> None:
    """Test score conversion when names map to multiple slugs."""
    # Set up test data where one name maps to multiple slugs
    choices = ["Shared Name", "Unique A", "Unique B"]
    name_to_slugs = {
        "Shared Name": ["slug1", "slug2"],  # Maps to both slugs
        "Unique A": ["slug1"],
        "Unique B": ["slug2"],
    }

    # Create a score matrix
    name_scores = np.array(
        [
            [50, 20, 30],  # Query scores for Shared Name, Unique A, Unique B
        ]
    )

    # Convert to slug scores
    slug_scores, slugs = _convert_name_scores_to_slug_scores(
        name_scores, choices, name_to_slugs
    )

    assert slugs == ["slug1", "slug2"]
    assert slug_scores.shape == (1, 2)

    # slug1 should get max(50, 20) = 50 (Shared Name and Unique A)
    # slug2 should get max(50, 30) = 50 (Shared Name and Unique B)
    assert slug_scores[0, 0] == 50  # slug1
    assert slug_scores[0, 1] == 50  # slug2


def test_convert_name_scores_edge_cases() -> None:
    """Test edge cases for the score conversion function."""
    # Edge case 1: Empty inputs
    name_scores = np.array([]).reshape(0, 0)
    slug_scores, slugs = _convert_name_scores_to_slug_scores(name_scores, [], {})
    assert slug_scores.shape == (0, 0)
    assert slugs == []

    # Edge case 2: Single name, single slug
    name_scores = np.array([[100]])  # 1 query, 1 name
    slug_scores, slugs = _convert_name_scores_to_slug_scores(
        name_scores, ["OnlyName"], {"OnlyName": ["only-slug"]}
    )
    assert slug_scores.shape == (1, 1)
    assert slugs == ["only-slug"]
    assert slug_scores[0, 0] == 100

    # Edge case 3: Multiple queries, no names/slugs
    name_scores = np.array([[], [], []]).reshape(3, 0)  # 3 queries, 0 names
    slug_scores, slugs = _convert_name_scores_to_slug_scores(name_scores, [], {})
    assert slug_scores.shape == (3, 0)
    assert slugs == []

    # Edge case 4: All names map to the same slug (complete deduplication)
    name_scores = np.array([[10, 20, 30, 40]])  # 1 query, 4 names
    name_to_slugs = {
        "Name1": ["the-only-slug"],
        "Name2": ["the-only-slug"],
        "Name3": ["the-only-slug"],
        "Name4": ["the-only-slug"],
    }
    slug_scores, slugs = _convert_name_scores_to_slug_scores(
        name_scores, ["Name1", "Name2", "Name3", "Name4"], name_to_slugs
    )
    assert slug_scores.shape == (1, 1)
    assert slugs == ["the-only-slug"]
    assert slug_scores[0, 0] == 40  # Max of all scores

    # Edge case 5: Scores of 0 (no similarity)
    name_scores = np.array([[0, 0], [0, 0]])  # 2 queries, 2 names, all zeros
    name_to_slugs = {"Name1": ["slug1"], "Name2": ["slug2"]}
    slug_scores, slugs = _convert_name_scores_to_slug_scores(
        name_scores, ["Name1", "Name2"], name_to_slugs
    )
    assert slug_scores.shape == (2, 2)
    assert np.all(slug_scores == 0)

    # Edge case 6: Very high dimensionality (many queries and names)
    num_queries, num_names = 100, 50
    name_scores = np.random.rand(num_queries, num_names) * 100
    names = [f"Name{i}" for i in range(num_names)]
    # Each name maps to its own slug
    name_to_slugs = {name: [f"slug{i}"] for i, name in enumerate(names)}

    slug_scores, slugs = _convert_name_scores_to_slug_scores(
        name_scores, names, name_to_slugs
    )
    assert slug_scores.shape == (num_queries, num_names)
    assert len(slugs) == num_names

    # Since each name maps to exactly one slug, but slugs are sorted,
    # we need to map scores correctly. For each slug, find its corresponding name.
    for slug_idx, slug in enumerate(slugs):
        # Find which name this slug came from (e.g., "slug5" came from "Name5")
        name_idx = int(slug.replace("slug", ""))
        # The scores for this slug should match the original name scores
        np.testing.assert_array_almost_equal(
            slug_scores[:, slug_idx], name_scores[:, name_idx]
        )


def test_convert_name_scores_missing_mappings() -> None:
    """Test behavior when some names don't have slug mappings."""
    # Set up test with some names missing from name_to_slugs
    name_scores = np.array([[10, 20, 30]])  # 1 query, 3 names
    choices = ["Name1", "Name2", "Name3"]
    name_to_slugs = {
        "Name1": ["slug1"],
        # "Name2" is missing - this could happen if there's a data inconsistency
        "Name3": ["slug3"],
    }

    # This should raise a KeyError because Name2 is not in name_to_slugs
    try:
        _convert_name_scores_to_slug_scores(name_scores, choices, name_to_slugs)
        raise AssertionError("Expected KeyError for missing name mapping")
    except KeyError as e:
        assert "Name2" in str(e)


def test_rank_slugs_by_query() -> None:
    """Test the _rank_slugs_by_query function."""
    # Test case: 2 queries, 4 slugs
    # Query 0 scores: [10, 30, 20, 40] -> should rank as [3, 1, 2, 0] (by score desc)
    # Query 1 scores: [50, 10, 30, 20] -> should rank as [0, 2, 3, 1]
    query_slug_scores = np.array(
        [
            [10, 30, 20, 40],  # Query 0: slug3=40, slug1=30, slug2=20, slug0=10
            [50, 10, 30, 20],  # Query 1: slug0=50, slug2=30, slug3=20, slug1=10
        ]
    )

    rankings = _rank_slugs_by_query(query_slug_scores)

    assert len(rankings) == 2  # One ranking per query
    assert len(rankings[0]) == 4  # All slugs ranked for query 0
    assert len(rankings[1]) == 4  # All slugs ranked for query 1

    # Query 0: best to worst should be [3, 1, 2, 0] (indices of slugs by score)
    assert rankings[0] == [3, 1, 2, 0]

    # Query 1: best to worst should be [0, 2, 3, 1]
    assert rankings[1] == [0, 2, 3, 1]


def test_rank_slugs_by_query_edge_cases() -> None:
    """Test edge cases for _rank_slugs_by_query."""
    # Empty case
    empty_scores = np.array([]).reshape(0, 0)
    rankings = _rank_slugs_by_query(empty_scores)
    assert rankings == []

    # Single query, single slug
    single_scores = np.array([[42]])
    rankings = _rank_slugs_by_query(single_scores)
    assert rankings == [[0]]

    # Tied scores - should be stable sort
    tied_scores = np.array([[10, 20, 20, 10]])  # Ties at 20 and 10
    rankings = _rank_slugs_by_query(tied_scores)
    # For ties, numpy.argsort is stable, so earlier indices come first
    assert rankings[0][:2] == [1, 2]  # Both score 20, but index 1 comes before 2
    assert rankings[0][2:] == [0, 3]  # Both score 10, but index 0 comes before 3


def test_reciprocal_rank_fusion() -> None:
    """Test the _reciprocal_rank_fusion function."""
    # Test case: 2 queries ranking 3 items differently
    # Query 0 ranking: [0, 1, 2] (slug 0 is best, slug 1 second, slug 2 third)
    # Query 1 ranking: [2, 0, 1] (slug 2 is best, slug 0 second, slug 1 third)
    rankings = [
        [0, 1, 2],  # Query 0
        [2, 0, 1],  # Query 1
    ]

    # Calculate expected RRF scores with k=60
    # Slug 0: 1/(60+1) + 1/(60+2) = 1/61 + 1/62 ≈ 0.01639 + 0.01613 ≈ 0.03252
    # Slug 1: 1/(60+2) + 1/(60+3) = 1/62 + 1/63 ≈ 0.01613 + 0.01587 ≈ 0.03200
    # Slug 2: 1/(60+3) + 1/(60+1) = 1/63 + 1/61 ≈ 0.01587 + 0.01639 ≈ 0.03226

    rrf_results = _reciprocal_rank_fusion(rankings, k=60)

    assert len(rrf_results) == 3
    # Results should be sorted by RRF score (best first)
    slug_indices = [result[0] for result in rrf_results]
    rrf_scores = [result[1] for result in rrf_results]

    # Slug 0 should have the highest RRF score
    assert slug_indices[0] == 0
    assert abs(rrf_scores[0] - (1 / 61 + 1 / 62)) < 1e-10

    # Check all scores are in descending order
    assert rrf_scores[0] > rrf_scores[1] > rrf_scores[2]


def test_reciprocal_rank_fusion_edge_cases() -> None:
    """Test edge cases for _reciprocal_rank_fusion."""
    # Empty rankings
    rrf_results = _reciprocal_rank_fusion([])
    assert rrf_results == []

    # Single ranking
    rrf_results = _reciprocal_rank_fusion([[0, 1, 2]])
    assert len(rrf_results) == 3
    # Should be in order of the single ranking
    assert [result[0] for result in rrf_results] == [0, 1, 2]

    # Rankings with different items (some items missing from some rankings)
    rankings = [
        [0, 1],  # Query 0 only sees slugs 0 and 1
        [2, 0, 3],  # Query 1 sees slugs 2, 0, and 3 (no slug 1)
    ]
    rrf_results = _reciprocal_rank_fusion(rankings)

    # All unique slugs should appear in results
    slug_indices = {result[0] for result in rrf_results}
    assert slug_indices == {0, 1, 2, 3}

    # Slug 0 appears in both rankings (ranks 1 and 2), should score well
    # Slug 1 only appears in first ranking (rank 2)
    # Slug 2 only appears in second ranking (rank 1)
    # Slug 3 only appears in second ranking (rank 3)


def test_search_wiki_name_to_slug_conversion(temp_db: SafeConnection) -> None:
    """Test that scores are correctly converted from names to slugs."""
    with temp_db.transaction_cursor() as cursor:
        # Set up test data
        chapter = Chapter.add_chapter(
            cursor, 1, ["Book 1", "Chapter 1"], "Chapter content..."
        )
        conversation = Conversation.create(cursor)
        chapter.set_conversation_id(conversation)

        # Create wiki pages with overlapping names
        blocks = []
        for i in range(2):
            block = Block.create_text(
                cursor,
                conversation.id,
                conversation.current_generation,
                "assistant",
                f"Creating page {i}",
            )
            blocks.append(block)

        # First character with multiple aliases
        blocks[0].write_wiki_page(
            chapter_id=1,
            slug="dragon-reborn",
            title="The Dragon Reborn",
            names=["Rand al'Thor", "Dragon Reborn", "Lord Dragon", "Car'a'carn"],
            summary="The prophesied savior",
            body="The one who will fight the Dark One",
        )

        # Second character with some similar names
        blocks[1].write_wiki_page(
            chapter_id=1,
            slug="false-dragon",
            title="False Dragon",
            names=["Logain Ablar", "False Dragon", "Dragon"],  # "Dragon" is ambiguous
            summary="A false Dragon",
            body="One who falsely claimed to be the Dragon",
        )

        # Search for "Dragon" - should match both pages but with different scores
        search_block = Block.create_tool_use(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            name="SearchWikiByName",
            use_id="search_dragon",
            params='{"names": ["Dragon"]}',
        )

        tool = SearchWikiByName(
            tool_id="search_dragon",
            tool_name="SearchWikiByName",
            names=["Dragon"],
            results_page=1,
        )

        # The search should work and both pages should get scores
        # The false-dragon page should get a perfect score (100) for exact match
        # The dragon-reborn page should get a high score for partial match
        tool.apply(search_block)

        # Check that we got a response using model method
        updated_search_block = Block.get_by_id(cursor, search_block.id)
        assert updated_search_block is not None
        assert updated_search_block.tool_response is not None
        assert updated_search_block.errored is False

        # Should contain search results with both characters
        response = updated_search_block.tool_response
        assert "Search Results" in response
        assert "dragon-reborn" in response or "false-dragon" in response


def test_search_wiki_pagination(temp_db: SafeConnection) -> None:
    """Test pagination functionality in search results."""
    with temp_db.transaction_cursor() as cursor:
        # Set up test data with enough pages to test pagination
        chapter = Chapter.add_chapter(
            cursor, 1, ["Book 1", "Chapter 1"], "Chapter content..."
        )
        conversation = Conversation.create(cursor)
        chapter.set_conversation_id(conversation)

        # Create 15 wiki pages (more than 1 page worth at size 10)
        for i in range(15):
            block = Block.create_text(
                cursor,
                conversation.id,
                conversation.current_generation,
                "assistant",
                f"Creating page {i}",
            )
            block.write_wiki_page(
                chapter_id=1,
                slug=f"character-{i:02d}",
                title=f"Character {i}",
                names=[f"Character {i}", f"Person {i}"],
                summary=f"Character number {i}",
                body=f"Details about character {i}",
            )

        from bookwiki.tools.wiki import SearchWikiByName

        # Test page 1 (should have 6 results)
        search_block1 = Block.create_tool_use(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            name="SearchWikiByName",
            use_id="search_page1",
            params='{"names": ["Character"], "results_page": 1}',
        )

        tool1 = SearchWikiByName(
            tool_id="search_page1",
            tool_name="SearchWikiByName",
            names=["Character"],
            results_page=1,
        )

        tool1.apply(search_block1)

        updated_search_block1 = Block.get_by_id(cursor, search_block1.id)
        assert updated_search_block1 is not None
        assert updated_search_block1.tool_response is not None
        response1 = updated_search_block1.tool_response

        assert "Search Results (Page 1" in response1
        assert "showing 6 of 15 total" in response1
        assert "1. Character" in response1  # Should have rank 1
        assert "6. Character" in response1  # Should have rank 6

        # Test page 2 (should have 6 results)
        search_block2 = Block.create_tool_use(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            name="SearchWikiByName",
            use_id="search_page2",
            params='{"names": ["Character"], "results_page": 2}',
        )

        tool2 = SearchWikiByName(
            tool_id="search_page2",
            tool_name="SearchWikiByName",
            names=["Character"],
            results_page=2,
        )

        tool2.apply(search_block2)

        updated_search_block2 = Block.get_by_id(cursor, search_block2.id)
        assert updated_search_block2 is not None
        assert updated_search_block2.tool_response is not None
        response2 = updated_search_block2.tool_response

        assert "Search Results (Page 2" in response2
        assert "showing 6 of 15 total" in response2
        assert "7. Character" in response2  # Should start at rank 7
        assert "12. Character" in response2  # Should end at rank 12

        # Test page 3 (should have 3 results)
        search_block3 = Block.create_tool_use(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            name="SearchWikiByName",
            use_id="search_page3",
            params='{"names": ["Character"], "results_page": 3}',
        )

        tool3 = SearchWikiByName(
            tool_id="search_page3",
            tool_name="SearchWikiByName",
            names=["Character"],
            results_page=3,
        )

        tool3.apply(search_block3)

        updated_search_block3 = Block.get_by_id(cursor, search_block3.id)
        assert updated_search_block3 is not None
        assert updated_search_block3.tool_response is not None
        response3 = updated_search_block3.tool_response

        assert "Search Results (Page 3" in response3
        assert "showing 3 of 15 total" in response3
        assert "13. Character" in response3  # Should start at rank 13
        assert "15. Character" in response3  # Should end at rank 15

        # Test page 4 (should have no results)
        search_block4 = Block.create_tool_use(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            name="SearchWikiByName",
            use_id="search_page4",
            params='{"names": ["Character"], "results_page": 4}',
        )

        tool4 = SearchWikiByName(
            tool_id="search_page4",
            tool_name="SearchWikiByName",
            names=["Character"],
            results_page=4,
        )

        tool4.apply(search_block4)

        updated_search_block4 = Block.get_by_id(cursor, search_block4.id)
        assert updated_search_block4 is not None
        assert updated_search_block4.tool_response is not None
        response4 = updated_search_block4.tool_response

        assert "No results found on page 4" in response4


def test_search_wiki_finds_pages_from_earlier_chapters(temp_db: SafeConnection) -> None:
    """Test that searching from a later chapter finds pages from earlier chapters."""
    with temp_db.transaction_cursor() as cursor:
        # Create pages in chapter 1
        chapter1 = Chapter.add_chapter(
            cursor, 1, ["Book 1", "Chapter 1"], "Chapter 1 content..."
        )
        conversation1 = Conversation.create(cursor)
        chapter1.set_conversation_id(conversation1)

        # Create wiki pages in chapter 1
        block1 = Block.create_text(
            cursor,
            conversation1.id,
            conversation1.current_generation,
            "assistant",
            "Creating page in chapter 1",
        )
        block1.write_wiki_page(
            chapter_id=1,
            slug="early-character",
            title="Early Character",
            names=["Early Character", "First Appearance"],
            summary="A character introduced in chapter 1",
            body="This character appears early in the story",
        )

        # Add more chapters
        chapter2 = Chapter.add_chapter(
            cursor, 2, ["Book 1", "Chapter 2"], "Chapter 2 content..."
        )
        conversation2 = Conversation.create(cursor)
        chapter2.set_conversation_id(conversation2)

        chapter3 = Chapter.add_chapter(
            cursor, 3, ["Book 1", "Chapter 3"], "Chapter 3 content..."
        )
        conversation3 = Conversation.create(cursor)
        chapter3.set_conversation_id(conversation3)

        # Create a page in chapter 3
        block3 = Block.create_text(
            cursor,
            conversation3.id,
            conversation3.current_generation,
            "assistant",
            "Creating page in chapter 3",
        )
        block3.write_wiki_page(
            chapter_id=3,
            slug="later-character",
            title="Later Character",
            names=["Later Character", "Third Chapter"],
            summary="A character introduced in chapter 3",
            body="This character appears later in the story",
        )

        # Simulate proper chapter progression: inherit pages from earlier chapters
        # Chapter 2: inherit from chapter 1
        WikiPage.copy_current_for_new_chapter(cursor, 2)

        # Chapter 3: inherit from chapter 2 (which includes chapter 1 pages)
        WikiPage.copy_current_for_new_chapter(cursor, 3)

        # Now search from chapter 3 context - should find both pages
        search_block = Block.create_tool_use(
            cursor=cursor,
            conversation_id=conversation3.id,
            generation=conversation3.current_generation,
            name="SearchWikiByName",
            use_id="search_all",
            params='{"names": ["Character"]}',
        )

        tool = SearchWikiByName(
            tool_id="search_all",
            tool_name="SearchWikiByName",
            names=["Character"],
            results_page=1,
        )

        tool.apply(search_block)

        updated_search_block = Block.get_by_id(cursor, search_block.id)
        assert updated_search_block is not None
        assert updated_search_block.tool_response is not None
        response = updated_search_block.tool_response

        # Should find both the early and later character
        assert "Search Results" in response
        assert "early-character" in response
        assert "later-character" in response
        assert "2 total" in response  # Should find exactly 2 pages

        # Now advance to chapter 5 and search again (simulating sequential processing)
        # Add intermediate chapter 4 to maintain sequential progression
        Chapter.add_chapter(cursor, 4, ["Book 1", "Chapter 4"], "Chapter 4 content...")
        WikiPage.copy_current_for_new_chapter(cursor, 4)

        chapter5 = Chapter.add_chapter(
            cursor, 5, ["Book 1", "Chapter 5"], "Chapter 5 content..."
        )
        conversation5 = Conversation.create(cursor)
        chapter5.set_conversation_id(conversation5)

        # Chapter 5: inherit from chapter 4 (includes all previous pages)
        WikiPage.copy_current_for_new_chapter(cursor, 5)

        search_block5 = Block.create_tool_use(
            cursor=cursor,
            conversation_id=conversation5.id,
            generation=conversation5.current_generation,
            name="SearchWikiByName",
            use_id="search_chapter5",
            params='{"names": ["Early"]}',
        )

        tool5 = SearchWikiByName(
            tool_id="search_chapter5",
            tool_name="SearchWikiByName",
            names=["Early"],
            results_page=1,
        )

        tool5.apply(search_block5)

        updated_search_block5 = Block.get_by_id(cursor, search_block5.id)
        assert updated_search_block5 is not None
        assert updated_search_block5.tool_response is not None
        response5 = updated_search_block5.tool_response

        # From chapter 5, should still find the page created in chapter 1
        assert "Search Results" in response5
        assert "early-character" in response5
        assert "Early Character" in response5


def test_read_wiki_page_no_chapters_started(temp_db: SafeConnection) -> None:
    """Test ReadWikiPage when no chapters have been started."""
    with temp_db.transaction_cursor() as cursor:
        # Add a chapter but don't start it
        Chapter.add_chapter(cursor, 1, ["Book 1", "Chapter 1"], "Content")
        conversation = Conversation.create(cursor)

        block = Block.create_tool_use(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            name="ReadWikiPage",
            use_id="read_no_started",
            params='{"slug": "test-page"}',
        )

        tool = ReadWikiPage(
            tool_id="read_no_started",
            tool_name="ReadWikiPage",
            slug="test-page",
        )

        with pytest.raises(LLMSolvableError, match="No chapters have been started yet"):
            tool._apply(block)


def test_write_wiki_page_no_chapters_started(temp_db: SafeConnection) -> None:
    """Test WriteWikiPage when no chapters have been started."""
    with temp_db.transaction_cursor() as cursor:
        # Add a chapter but don't start it
        Chapter.add_chapter(cursor, 1, ["Book 1", "Chapter 1"], "Content")
        conversation = Conversation.create(cursor)

        block = Block.create_tool_use(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            name="WriteWikiPage",
            use_id="write_no_started",
            params=(
                '{"slug": "new-page", "title": "New Page", "names": ["New"], '
                '"summary": "New page", "body": "Content", "create": true}'
            ),
        )

        tool = WriteWikiPage(
            tool_id="write_no_started",
            tool_name="WriteWikiPage",
            slug="new-page",
            title="New Page",
            names=["New"],
            summary="New page",
            body="Content",
            create=True,
        )

        with pytest.raises(LLMSolvableError, match="No chapters have been started yet"):
            tool._apply(block)


def test_search_wiki_by_name_no_chapters_started(temp_db: SafeConnection) -> None:
    """Test SearchWikiByName when no chapters have been started."""
    with temp_db.transaction_cursor() as cursor:
        # Add a chapter but don't start it
        Chapter.add_chapter(cursor, 1, ["Book 1", "Chapter 1"], "Content")
        conversation = Conversation.create(cursor)

        block = Block.create_tool_use(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            name="SearchWikiByName",
            use_id="search_no_started",
            params='{"names": ["Test Character"]}',
        )

        tool = SearchWikiByName(
            tool_id="search_no_started",
            tool_name="SearchWikiByName",
            names=["Test Character"],
            results_page=None,
        )

        with pytest.raises(LLMSolvableError, match="No chapters have been started yet"):
            tool._apply(block)


def test_search_wiki_by_name_high_page_number(temp_db: SafeConnection) -> None:
    """Test SearchWikiByName with high page numbers (no results)."""
    with temp_db.transaction_cursor() as cursor:
        chapter = Chapter.add_chapter(cursor, 1, ["Book 1", "Chapter 1"], "Content")
        conversation = Conversation.create(cursor)
        chapter.set_conversation_id(conversation)

        # Create a wiki page so there is some data
        create_block = Block.create_text(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            role="assistant",
            text="Creating page",
        )
        create_block.write_wiki_page(
            chapter_id=chapter.id,
            slug="test-character",
            title="Test Character",
            names=["Test Character"],
            summary="A test character",
            body="A test character for testing.",
        )

        # Search with high page number (should get no results on this page)
        block = Block.create_tool_use(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            name="SearchWikiByName",
            use_id="search_page_high",
            params='{"names": ["Test Character"], "results_page": 100}',
        )

        tool = SearchWikiByName(
            tool_id="search_page_high",
            tool_name="SearchWikiByName",
            names=["Test Character"],
            results_page=100,
        )

        tool._apply(block)

        # Check response indicates no results on this page
        updated_block = Block.get_by_id(cursor, block.id)
        assert updated_block is not None
        assert updated_block.tool_response is not None
        assert "No results found on page 100" in updated_block.tool_response


def test_read_wiki_page_with_suggestions(temp_db: SafeConnection) -> None:
    """Test ReadWikiPage nonexistent page with similar page suggestions."""
    with temp_db.transaction_cursor() as cursor:
        chapter = Chapter.add_chapter(cursor, 1, ["Book 1", "Chapter 1"], "Content")
        conversation = Conversation.create(cursor)
        chapter.set_conversation_id(conversation)

        # Create a similar page
        create_block = Block.create_text(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            role="assistant",
            text="Creating page",
        )
        create_block.write_wiki_page(
            chapter_id=chapter.id,
            slug="gandalf-the-grey",
            title="Gandalf the Grey",
            names=["Gandalf the Grey"],
            summary="A powerful wizard",
            body="A powerful wizard who helps the Fellowship.",
        )

        # Try to read a similar but wrong slug
        block = Block.create_tool_use(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            name="ReadWikiPage",
            use_id="read_similar",
            params='{"slug": "gandalf-grey"}',  # Similar but not exact
        )

        tool = ReadWikiPage(
            tool_id="read_similar",
            tool_name="ReadWikiPage",
            slug="gandalf-grey",
        )

        # Should raise an error with suggestions
        with pytest.raises(LLMSolvableError) as exc_info:
            tool._apply(block)

        error_msg = str(exc_info.value)
        assert "No page exists with slug 'gandalf-grey'" in error_msg
        # The exact suggestion format may vary based on implementation
        # but should include the similar page
        assert "gandalf-the-grey" in error_msg or "Gandalf the Grey" in error_msg
