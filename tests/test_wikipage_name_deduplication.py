"""Tests for WikiPage name deduplication functionality."""

from typing import Any

import pytest

from bookwiki.models.chapter import Chapter
from bookwiki.models.conversation import Conversation
from bookwiki.models.wikipage import (
    WikiPage,
    _deduplicate_names,
    _normalize_name_key,
    _select_best_name_from_group,
)


class TestNameNormalization:
    """Test the name key normalization function."""

    def test_removes_non_alphanumeric(self) -> None:
        """Test removal of non-alphanumeric characters."""
        assert _normalize_name_key("Harry Potter!") == "harry potter"
        assert _normalize_name_key("Harry-Potter") == "harry potter"
        assert _normalize_name_key("Harry.Potter") == "harry potter"
        assert _normalize_name_key("Harry@#$Potter") == "harry potter"

    def test_normalizes_whitespace(self) -> None:
        """Test whitespace normalization."""
        assert _normalize_name_key("Harry  Potter") == "harry potter"
        assert _normalize_name_key("  Harry Potter  ") == "harry potter"
        assert _normalize_name_key("Harry\tPotter") == "harry potter"
        assert _normalize_name_key("Harry\n\rPotter") == "harry potter"

    def test_converts_to_lowercase(self) -> None:
        """Test case conversion."""
        assert _normalize_name_key("HARRY POTTER") == "harry potter"
        assert _normalize_name_key("Harry Potter") == "harry potter"
        assert _normalize_name_key("harry potter") == "harry potter"
        assert _normalize_name_key("HaRrY PoTtEr") == "harry potter"

    def test_removes_leading_the(self) -> None:
        """Test removal of leading 'the '."""
        assert _normalize_name_key("The Harry Potter") == "harry potter"
        assert _normalize_name_key("the harry potter") == "harry potter"
        assert _normalize_name_key("THE HARRY POTTER") == "harry potter"
        assert (
            _normalize_name_key("Theodore Roosevelt") == "theodore roosevelt"
        )  # 'the' not at start

    def test_complex_cases(self) -> None:
        """Test complex combinations of normalization rules."""
        assert _normalize_name_key("The  Harry--Potter!!!") == "harry potter"
        assert _normalize_name_key("  THE   HARRY   POTTER  ") == "harry potter"
        assert _normalize_name_key("The-Boy-Who-Lived") == "boy who lived"

    def test_edge_cases(self) -> None:
        """Test edge cases."""
        assert _normalize_name_key("") == ""
        assert _normalize_name_key("   ") == ""
        assert _normalize_name_key("!!!") == ""
        assert _normalize_name_key("The") == "the"  # Just "the" stays as "the"
        assert _normalize_name_key("The ") == "the"  # "The " normalizes to "the"


class TestSelectBestName:
    """Test the best name selection function."""

    def test_prefers_longest_name(self) -> None:
        """Test that longer names are preferred."""
        names = ["Harry", "Harry Potter", "Harry P"]
        assert _select_best_name_from_group(names) == "Harry Potter"

    def test_prefers_more_uppercase(self) -> None:
        """Test names with more uppercase letters preferred when same length."""
        names = ["harry potter", "Harry Potter", "HARRY POTTER"]
        # All same length, so most uppercase wins
        assert _select_best_name_from_group(names) == "HARRY POTTER"

    def test_lexicographic_tiebreaker(self) -> None:
        """Test lexicographic ordering as final tiebreaker."""
        names = ["Harry Potter", "Harry Aotter"]  # Same length, same uppercase
        assert _select_best_name_from_group(names) == "Harry Potter"

    def test_combined_criteria(self) -> None:
        """Test that criteria are applied in correct order."""
        # Length takes precedence
        names = ["HARRY", "Harry Potter"]  # First has more uppercase, second is longer
        assert _select_best_name_from_group(names) == "Harry Potter"

        # Uppercase takes precedence when same length
        names = ["harry p", "Harry P", "HARRY P"]
        assert _select_best_name_from_group(names) == "HARRY P"

    def test_single_name(self) -> None:
        """Test with single name in group."""
        assert _select_best_name_from_group(["Harry"]) == "Harry"

    def test_empty_names_handled(self) -> None:
        """Test that empty names are handled."""
        names = ["", "Harry"]
        assert _select_best_name_from_group(names) == "Harry"


class TestDeduplicateNames:
    """Test the full name deduplication function."""

    def test_basic_deduplication(self) -> None:
        """Test basic deduplication of similar names."""
        names = ["Harry Potter", "harry potter", "HARRY POTTER"]
        result = _deduplicate_names(names)
        assert len(result) == 1
        assert result[0] == "HARRY POTTER"  # Most uppercase wins

    def test_the_prefix_deduplication(self) -> None:
        """Test deduplication with 'the' prefix variations."""
        names = ["The Boy Who Lived", "Boy Who Lived", "the boy who lived"]
        result = _deduplicate_names(names)
        assert len(result) == 1
        assert result[0] == "The Boy Who Lived"  # Longest wins

    def test_punctuation_variations(self) -> None:
        """Test deduplication with punctuation variations."""
        names = ["Harry Potter", "Harry-Potter", "Harry.Potter", "Harry, Potter"]
        result = _deduplicate_names(names)
        assert len(result) == 1
        # All normalize to same key "harry potter"
        # "Harry, Potter" has the most characters and should win
        assert result[0] == "Harry, Potter"

    def test_whitespace_variations(self) -> None:
        """Test deduplication with whitespace variations."""
        names = ["Harry  Potter", "  Harry Potter  ", "Harry\tPotter", "Harry Potter"]
        result = _deduplicate_names(names)
        assert len(result) == 1

    def test_multiple_distinct_names(self) -> None:
        """Test that distinct names are preserved."""
        names = ["Harry Potter", "Hermione Granger", "Ron Weasley"]
        result = _deduplicate_names(names)
        assert len(result) == 3
        assert set(result) == set(names)

    def test_mixed_duplicates_and_distinct(self) -> None:
        """Test mix of duplicates and distinct names."""
        names = [
            "Harry Potter",
            "harry potter",  # Duplicates
            "Hermione Granger",  # Distinct
            "Ron Weasley",
            "ron weasley",  # Duplicates
        ]
        result = _deduplicate_names(names)
        assert len(result) == 3
        # Check that we got one from each group
        assert "Harry Potter" in result  # Uppercase wins
        assert "Hermione Granger" in result
        assert "Ron Weasley" in result  # Uppercase wins

    def test_empty_list(self) -> None:
        """Test with empty list."""
        assert _deduplicate_names([]) == []

    def test_all_normalize_to_empty(self) -> None:
        """Test when all names normalize to empty strings."""
        names = ["!!!", "   ", "---"]
        result = _deduplicate_names(names)
        assert len(result) == 1
        assert result[0] == "!!!"  # First original name kept

    def test_sorting_output(self) -> None:
        """Test that output is sorted."""
        names = ["Zebra", "Apple", "Banana"]
        result = _deduplicate_names(names)
        assert result == ["Apple", "Banana", "Zebra"]


class TestWikiPageCreateWithDeduplication:
    """Test WikiPage.create() with name deduplication."""

    @pytest.fixture
    def setup_db(self, temp_db: Any) -> tuple[Any, int]:
        """Set up database with basic data."""
        with temp_db.transaction_cursor() as cursor:
            # Create a chapter
            Chapter.add_chapter(cursor, 1, ["Chapter 1"], "Content")

            # Create a conversation and block
            conv = Conversation.create(cursor)
            block = conv.add_user_text("Test")
            block_id = block.id

        return temp_db, block_id

    def test_create_with_duplicate_names(self, setup_db: tuple[Any, int]) -> None:
        """Test creating wiki page with duplicate names."""
        db, block_id = setup_db

        with db.transaction_cursor() as cursor:
            # Create wiki page with duplicate names
            names = ["Harry Potter", "harry potter", "HARRY POTTER", "Harry-Potter"]
            page = WikiPage.create(
                cursor=cursor,
                chapter_id=1,
                slug="harry-potter",
                create_block_id=block_id,
                title="Harry Potter",
                names=names,
                summary="The Boy Who Lived",
                body="Harry Potter is a wizard.",
            )

            # Check that duplicates were removed
            assert len(page.names) == 1
            assert page.names[0] == "HARRY POTTER"  # Most uppercase wins

            # Verify in database
            name_rows = cursor.execute(
                """SELECT wn.name FROM wiki_name wn
                   JOIN wiki_page_name wpn ON wn.id = wpn.wiki_name_id
                   WHERE wpn.wiki_page_id = ?""",
                (page.id,),
            ).fetchall()

            assert len(name_rows) == 1
            assert name_rows[0]["name"] == "HARRY POTTER"

    def test_create_with_mixed_names(self, setup_db: tuple[Any, int]) -> None:
        """Test creating wiki page with mix of duplicate and distinct names."""
        db, block_id = setup_db

        with db.transaction_cursor() as cursor:
            names = [
                "Harry Potter",
                "harry potter",  # Duplicates
                "The Boy Who Lived",
                "Boy Who Lived",  # Duplicates with 'the'
                "The Chosen One",  # Distinct
            ]

            page = WikiPage.create(
                cursor=cursor,
                chapter_id=1,
                slug="harry-potter",
                create_block_id=block_id,
                title="Harry Potter",
                names=names,
                summary="The Boy Who Lived",
                body="Harry Potter is a wizard.",
            )

            # Should have 3 deduplicated names
            assert len(page.names) == 3

            # Verify the deduplicated names
            expected_names = {"Harry Potter", "The Boy Who Lived", "The Chosen One"}
            assert set(page.names) == expected_names

    def test_create_requires_names(self, setup_db: tuple[Any, int]) -> None:
        """Test that creating wiki page without names raises error."""
        db, block_id = setup_db

        with (
            db.transaction_cursor() as cursor,
            pytest.raises(ValueError, match="must have at least one name"),
        ):
            WikiPage.create(
                cursor=cursor,
                chapter_id=1,
                slug="test",
                create_block_id=block_id,
                title="Test",
                names=[],  # Empty names list
                summary="Test",
                body="Test",
            )

    def test_update_with_deduplication(self, setup_db: tuple[Any, int]) -> None:
        """Test updating a wiki page with new names also deduplicates."""
        db, block_id = setup_db

        with db.transaction_cursor() as cursor:
            # Create initial page
            WikiPage.create(
                cursor=cursor,
                chapter_id=1,
                slug="harry-potter",
                create_block_id=block_id,
                title="Harry Potter",
                names=["Harry"],
                summary="A wizard",
                body="Content 1",
            )

            # Create new block for update
            conv2 = Conversation.create(cursor)
            block2 = conv2.add_user_text("Update")

            # Update with duplicate names
            page2 = WikiPage.create(
                cursor=cursor,
                chapter_id=1,
                slug="harry-potter",
                create_block_id=block2.id,
                title="Harry Potter",
                names=[
                    "Harry Potter",
                    "harry potter",
                    "Harry-Potter",
                    "The Boy Who Lived",
                ],
                summary="The Boy Who Lived",
                body="Content 2",
            )

            # Check deduplication on update
            # "Harry-Potter" wins over "Harry Potter" due to lexicographic ordering
            expected_names = {"Harry-Potter", "The Boy Who Lived"}
            assert len(page2.names) == 2
            assert set(page2.names) == expected_names
