from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from functools import cached_property
from sqlite3 import Cursor, Row
from typing import TYPE_CHECKING

from bookwiki.utils import utc_now

if TYPE_CHECKING:
    from .block import Block
    from .chapter import Chapter

logger = logging.getLogger(__name__)


def _normalize_name_key(name: str) -> str:
    """Normalize a name to create a grouping key.

    Args:
        name: Name to normalize

    Returns:
        Normalized key for grouping similar names

    Example:
        "The  Harry Potter!" -> "harry potter"
        "harry-potter" -> "harry potter"
    """
    # Replace non-word characters (except spaces) with spaces
    # This preserves Unicode letters while removing punctuation
    # \w matches [a-zA-Z0-9_] plus Unicode letter characters
    key = re.sub(r"[^\w\s]+", " ", name, flags=re.UNICODE)
    # Also replace underscores with spaces since \w includes them
    key = key.replace("_", " ")

    # Normalize whitespace: trim and replace multiple spaces with single space
    key = " ".join(key.split())

    # Convert to lowercase
    key = key.lower()

    # Remove leading "the " only if there's more content after it
    if key.startswith("the ") and len(key) > 4:
        key = key[4:]

    return key


def _select_best_name_from_group(names: list[str]) -> str:
    """Select the best name from a group of similar names.

    Selection criteria (in order):
    1. Longest name
    2. Most uppercase letters
    3. Lexicographically first

    Args:
        names: List of similar names to choose from

    Returns:
        The best name from the group
    """

    def count_uppercase(s: str) -> int:
        return sum(1 for c in s if c.isupper())

    # Sort by: length (desc), uppercase count (desc), name (asc for tiebreaker)
    return max(names, key=lambda n: (len(n), count_uppercase(n), n))


def _deduplicate_names(names: list[str]) -> list[str]:
    """Deduplicate a list of names based on normalized keys.

    Groups names by their normalized key, then selects the best
    name from each group.

    Args:
        names: List of names to deduplicate

    Returns:
        Deduplicated list of names
    """
    if not names:
        return []

    # Group names by normalized key
    groups = defaultdict(list)
    for name in names:
        key = _normalize_name_key(name)
        # Skip empty keys (names that become empty after normalization)
        if key:
            groups[key].append(name)

    # Select best name from each group
    deduplicated = []
    for group_names in groups.values():
        if group_names:
            deduplicated.append(_select_best_name_from_group(group_names))

    # If all names normalized to empty strings, keep at least one original name
    if not deduplicated and names:
        deduplicated = [names[0]]

    return sorted(deduplicated)  # Sort for consistent ordering


def _convert_values_to_ranks(values: list[int]) -> list[int]:
    """Convert a list of values to ranks, handling ties properly.

    Args:
        values: List of values to rank (higher values get better ranks)

    Returns:
        List of ranks where rank 1 = best (highest value), ties get same rank

    Example:
        values=[3, 3, 2] -> ranks=[1, 1, 3] (both 3's get rank 1, next gets rank 3)
    """
    if not values:
        return []

    # Create (value, original_index) pairs and sort by value descending
    indexed_values = [(val, i) for i, val in enumerate(values)]
    indexed_values.sort(key=lambda x: x[0], reverse=True)

    # Assign ranks, handling ties
    ranks = [0] * len(values)
    current_rank = 1

    for i, (value, original_index) in enumerate(indexed_values):
        if i > 0 and indexed_values[i - 1][0] != value:
            # Value changed, update rank to current position + 1
            current_rank = i + 1
        ranks[original_index] = current_rank

    return ranks


def _wiki_page_reciprocal_rank_fusion(
    pages_with_ranks: list[tuple["WikiPage", int, int]], k: int = 60
) -> list["WikiPage"]:
    """Combine WikiPage rankings using Reciprocal Rank Fusion.

    Args:
        pages_with_ranks: List of (page, latest_rank, frequency_rank) tuples
            where ranks are 1-based (1 = best, 2 = second best, etc.)
        k: RRF parameter (typically 60), controls the importance of top ranks

    Returns:
        List of WikiPage objects sorted by RRF score (best first),
        with ties broken by body length (longer first)
    """
    if not pages_with_ranks:
        return []

    # Calculate RRF score for each page
    rrf_scores = []
    for page, latest_rank, frequency_rank in pages_with_ranks:
        # RRF formula: RRF(d) = Σ(1 / (k + rank(d, q))) for all queries q
        rrf_score = (1.0 / (k + latest_rank)) + (1.0 / (k + frequency_rank))
        rrf_scores.append((page, rrf_score))

    # Sort by RRF score (descending = best first), break ties by body length
    sorted_results = sorted(
        rrf_scores,
        key=lambda x: (x[1], len(x[0].body)),
        reverse=True,
    )

    return [page for page, _ in sorted_results]


@dataclass(frozen=True)
class WikiPage:
    _cursor: Cursor
    id: int
    chapter_id: int
    slug: str
    create_time: datetime
    create_block_id: int
    title: str
    names: list[str]
    summary: str
    body: str

    @staticmethod
    def get_by_id(cursor: Cursor, page_id: int) -> "WikiPage | None":
        """Get a WikiPage by its ID."""
        row = cursor.execute(
            "SELECT * FROM wiki_page WHERE id = ?", (page_id,)
        ).fetchone()

        if row is None:
            return None

        return WikiPage._from_row(cursor, row)

    @staticmethod
    def get_by_create_block_id(
        cursor: Cursor, create_block_id: int
    ) -> "WikiPage | None":
        """Get a WikiPage by its create_block ID."""
        row = cursor.execute(
            "SELECT * FROM wiki_page WHERE create_block = ?", (create_block_id,)
        ).fetchone()

        if row is None:
            return None

        return WikiPage._from_row(cursor, row)

    @staticmethod
    def create(
        cursor: Cursor,
        chapter_id: int,
        slug: str,
        create_block_id: int,
        title: str,
        names: list[str],
        summary: str,
        body: str,
    ) -> "WikiPage":
        now = utc_now()
        if not names:
            raise ValueError("Wiki pages must have at least one name")

        cursor.execute(
            """INSERT INTO wiki_page (chapter, slug, create_time, create_block,
               title, summary, body) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (chapter_id, slug, now.isoformat(), create_block_id, title, summary, body),
        )
        wiki_page_id = cursor.lastrowid
        assert wiki_page_id is not None

        # Deduplicate names to avoid storing similar variations
        names = _deduplicate_names(names)

        for name in names:
            # Use RETURNING clause or lastrowid to avoid the second SELECT
            cursor.execute("INSERT OR IGNORE INTO wiki_name (name) VALUES (?)", (name,))
            # Get the ID either from the insert or from existing row
            cursor.execute("SELECT id FROM wiki_name WHERE name = ?", (name,))
            wiki_name_id = cursor.fetchone()[0]
            cursor.execute(
                "INSERT INTO wiki_page_name (wiki_page_id, wiki_name_id) VALUES (?, ?)",
                (wiki_page_id, wiki_name_id),
            )

        # Update wiki_page_current table
        if title == "":
            # If title is explicitly empty, remove from wiki_page_current (deletion)
            cursor.execute(
                "DELETE FROM wiki_page_current WHERE chapter = ? AND slug = ?",
                (chapter_id, slug),
            )
        else:
            # Upsert the current wiki page for this chapter and slug
            # This includes whitespace-only titles, which should be preserved
            cursor.execute(
                """INSERT INTO wiki_page_current (chapter, slug, wiki_page)
                   VALUES (?, ?, ?)
                   ON CONFLICT(chapter, slug)
                   DO UPDATE SET wiki_page = excluded.wiki_page""",
                (chapter_id, slug, wiki_page_id),
            )

        return WikiPage(
            _cursor=cursor,
            id=wiki_page_id,
            chapter_id=chapter_id,
            slug=slug,
            create_time=now,
            create_block_id=create_block_id,
            title=title,
            names=names,
            summary=summary,
            body=body,
        )

    @staticmethod
    def copy_current_for_new_chapter(cursor: Cursor, chapter_id: int) -> None:
        """Copy all current wiki pages from the previous chapter to the new chapter.

        This ensures that all wiki pages visible in chapter N-1 are also visible
        in chapter N unless explicitly overridden by new pages.
        """
        cursor.execute(
            """INSERT INTO wiki_page_current (chapter, slug, wiki_page)
               SELECT ?, slug, wiki_page
               FROM wiki_page_current
               WHERE chapter = ?""",
            (chapter_id, chapter_id - 1),
        )

    @staticmethod
    def read_page_at(cursor: Cursor, slug: str, chapter: int) -> "WikiPage" | None:
        row = cursor.execute(
            """SELECT wp.* FROM wiki_page_current wpc
               JOIN wiki_page wp ON wpc.wiki_page = wp.id
               WHERE wpc.chapter = ? AND wpc.slug = ?""",
            (chapter, slug),
        ).fetchone()

        if row is None:
            return None

        return WikiPage._from_row(cursor, row)

    @staticmethod
    def get_all_slugs(cursor: Cursor, chapter_id: int) -> set[str]:
        """Get all unique slugs that exist at the given chapter.

        Args:
            cursor: Database cursor
            chapter_id: Chapter ID to get slugs up to

        Returns:
            Set of unique slug strings
        """
        rows = cursor.execute(
            """SELECT slug
               FROM wiki_page_current
               WHERE chapter = ?""",
            (chapter_id,),
        ).fetchall()

        return {row["slug"] for row in rows}

    @staticmethod
    def get_name_slug_pairs(cursor: Cursor, chapter: int) -> list[tuple[str, str]]:
        """Get all unique (name, slug) pairs for wiki pages up to the given chapter."""
        rows = cursor.execute(
            """SELECT DISTINCT wn.name, wp.slug
               FROM wiki_page_current wpc
               JOIN wiki_page wp ON wpc.wiki_page = wp.id
               JOIN wiki_page_name wpn ON wp.id = wpn.wiki_page_id
               JOIN wiki_name wn ON wpn.wiki_name_id = wn.id
               WHERE wpc.chapter = ?
               ORDER BY wn.name, wp.slug""",
            (chapter,),
        ).fetchall()

        return [(row["name"], row["slug"]) for row in rows]

    @staticmethod
    def get_all_pages_chapter(cursor: Cursor, chapter_id: int) -> list["WikiPage"]:
        """Get all wiki pages visible at the given chapter (latest of each slug)."""
        # TODO: this should be paginated eventually.
        rows = cursor.execute(
            """SELECT wp.*, COUNT(DISTINCT wp2.chapter) as distinct_chapters
               FROM wiki_page_current wpc
               JOIN wiki_page wp ON wpc.wiki_page = wp.id
               JOIN wiki_page wp2 ON wp.slug = wp2.slug AND wp2.chapter <= ?
               WHERE wpc.chapter = ?
               GROUP BY wp.id
               ORDER BY wp.title""",
            (chapter_id, chapter_id),
        ).fetchall()

        if not rows:
            return []

        # Create WikiPage objects with their ranking criteria
        pages_with_data = []
        for row in rows:
            page = WikiPage._from_row(cursor, row)
            pages_with_data.append((page, row["distinct_chapters"]))

        # Extract values for ranking
        chapter_ids = [page.chapter_id for page, _ in pages_with_data]
        distinct_chapters = [distinct_count for _, distinct_count in pages_with_data]

        # Convert values to proper ranks (ties get same rank)
        latest_ranks = _convert_values_to_ranks(chapter_ids)
        frequency_ranks = _convert_values_to_ranks(distinct_chapters)

        # Create input for RRF: (page, latest_rank, frequency_rank) tuples
        pages_with_ranks = []
        for i, (page, _) in enumerate(pages_with_data):
            pages_with_ranks.append((page, latest_ranks[i], frequency_ranks[i]))

        # Use RRF to get final ordering
        return _wiki_page_reciprocal_rank_fusion(pages_with_ranks)

    @staticmethod
    def get_versions_by_slug(
        cursor: Cursor, slug: str, chapter: int
    ) -> list["WikiPage"]:
        """Get all versions of a wiki page by slug up to given chapter.

        Results are ordered by chapter/create_time ascending.
        """
        rows = cursor.execute(
            """SELECT * FROM wiki_page
               WHERE slug = ? AND chapter <= ?
               ORDER BY chapter ASC, create_time ASC""",
            (slug, chapter),
        ).fetchall()

        versions = []
        for row in rows:
            versions.append(WikiPage._from_row(cursor, row))

        return versions

    @cached_property
    def create_block(self) -> "Block":
        """Get the block that created this wiki page."""
        from .block import Block

        block = Block.get_by_id(self._cursor, self.create_block_id)
        if block is None:
            raise ValueError(
                f"WikiPage {self.id} references non-existent create_block "
                f"{self.create_block_id}"
            )
        return block

    @cached_property
    def chapter(self) -> "Chapter":
        from .chapter import Chapter

        chapter = Chapter.read_chapter(self._cursor, self.chapter_id)

        if chapter is None:
            raise ValueError(
                f"WikiPage {self.id} references non-existent chapter{self.chapter_id}"
            )

        return chapter

    @cached_property
    def first_chapter(self) -> "Chapter":
        from .chapter import Chapter

        row = self._cursor.execute(
            """SELECT MIN(chapter) as first_chapter_id
               FROM wiki_page
               WHERE slug = ?""",
            (self.slug,),
        ).fetchone()

        if row is None or row["first_chapter_id"] is None:
            raise ValueError(f"No chapters found for slug '{self.slug}'")

        first_chapter_id = row["first_chapter_id"]
        chapter = Chapter.read_chapter(self._cursor, first_chapter_id)

        if chapter is None:
            raise ValueError(
                f"WikiPage {self.id} references non-existent first chapter "
                f"{first_chapter_id}"
            )

        return chapter

    @staticmethod
    def _from_row(cursor: Cursor, row: Row) -> "WikiPage":
        """Create a WikiPage instance from a database row.

        Args:
            cursor: Database cursor
            row: Database row containing wiki_page data
        """
        name_rows = cursor.execute(
            """SELECT wn.name FROM wiki_name wn
               JOIN wiki_page_name wpn ON wn.id = wpn.wiki_name_id
               WHERE wpn.wiki_page_id = ?""",
            (row["id"],),
        ).fetchall()

        names = [name_row["name"] for name_row in name_rows]

        return WikiPage(
            _cursor=cursor,
            id=row["id"],
            chapter_id=row["chapter"],
            slug=row["slug"],
            create_time=datetime.fromisoformat(row["create_time"]),
            create_block_id=row["create_block"],
            title=row["title"],
            names=names,
            summary=row["summary"],
            body=row["body"],
        )

    def delete_and_redirect(
        self, block: "Block", redirect_to: str
    ) -> tuple[list["WikiPage"], str]:
        """Delete this wiki page and redirect all links pointing to it.

        Args:
            block: The block to use for writing updated pages
            redirect_to: The slug to redirect links to (empty string removes links)

        Returns:
            Tuple of (pages_updated, response_message)
        """

        cursor = self._cursor

        # Find all pages that link to this page
        pages_to_update = self._find_pages_with_links_to(
            cursor, self.slug, self.chapter_id
        )

        # Update all pages that have links to the deleted page
        for page in pages_to_update:
            updated_body = self._replace_links_in_body(
                page.body, self.slug, redirect_to
            )

            block.write_wiki_page(
                self.chapter_id,
                page.slug,
                page.title,
                page.names,
                page.summary,
                updated_body,
            )

        # Delete this page by writing empty content
        block.write_wiki_page(
            self.chapter_id,
            self.slug,
            "",
            [""],
            "",
            "",
        )

        # Create response message
        if redirect_to == "":
            response = f"Wiki page '{self.slug}' deleted and all links removed."
        else:
            response = (
                f"Wiki page '{self.slug}' deleted and redirected to '{redirect_to}'."
            )

        if pages_to_update:
            response += (
                f" Updated {len(pages_to_update)} page(s) with redirected links."
            )

        return pages_to_update, response

    def _find_pages_with_links_to(
        self, cursor: Cursor, target_slug: str, chapter_id: int
    ) -> list["WikiPage"]:
        """Find all wiki pages that contain links to the target slug.

        Args:
            cursor: Database cursor
            target_slug: The slug to search for in wiki links
            chapter_id: The chapter ID

        Returns:
            List of WikiPage objects that contain links to the target slug
        """
        from bookwiki.utils import extract_wiki_links

        all_pages = WikiPage.get_all_pages_chapter(cursor, chapter_id)
        pages_with_links = []

        for page in all_pages:
            # Skip the page being deleted
            if page.slug == target_slug:
                continue

            # Extract links from the page body
            wiki_links = extract_wiki_links(page.body)

            # Check if any link points to our target slug
            if any(link.slug == target_slug for link in wiki_links):
                pages_with_links.append(page)

        return pages_with_links

    def _replace_links_in_body(self, body: str, old_slug: str, new_slug: str) -> str:
        """Replace all wiki links pointing to old_slug with new_slug.

        Args:
            body: The body content containing wiki links
            old_slug: The slug to replace
            new_slug: The slug to replace with (empty string removes the link)

        Returns:
            Updated body with replaced or removed links
        """
        from bookwiki.utils import extract_wiki_links

        # Extract all wiki links
        wiki_links = extract_wiki_links(body)

        # Process each link that points to the old slug
        updated_body = body
        for link in wiki_links:
            if link.slug == old_slug:
                # Create the old link pattern
                old_link = f"[{link.display_text}]({link.target})"

                if new_slug == "":
                    # Remove the link entirely, keeping just the display text
                    new_content = link.display_text
                else:
                    # Construct the new target, preserving any path prefix
                    if "/" in link.target:
                        # Replace just the slug part (after the last /)
                        prefix = "/".join(link.target.split("/")[:-1])
                        new_target = f"{prefix}/{new_slug}"
                    else:
                        new_target = new_slug
                    new_content = f"[{link.display_text}]({new_target})"

                # Replace in the body
                updated_body = updated_body.replace(old_link, new_content)

        return updated_body
