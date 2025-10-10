from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from functools import cached_property
from sqlite3 import Cursor, IntegrityError, Row

# Import TYPE_CHECKING to avoid circular imports
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .conversation import Conversation
    from .wikipage import WikiPage

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChapterName:
    id: int
    name: list[str]

    @property
    def url_id(self) -> int:
        """Return the 1-based identifier used in URLs."""
        return self.id + 1


@dataclass(frozen=True)
class Chapter:
    _cursor: Cursor
    id: int
    name: list[str]  # Stored as JSON in database
    text: str
    conversation_id: Optional[int]  # FK to conversation.id
    chapter_summary_page_id: Optional[int]  # FK to wiki_page.id

    @staticmethod
    def add_chapter(
        cursor: Cursor, chapter_id: int, name: list[str], text: str
    ) -> "Chapter":
        # Validate that name list is not empty
        if not name:
            raise ValueError("Chapter name list cannot be empty")

        name_json = json.dumps(name)
        try:
            cursor.execute(
                """INSERT INTO chapter (
                    id, name, text, conversation_id, chapter_summary_page_id
                ) VALUES (?, ?, ?, ?, ?)""",
                (chapter_id, name_json, text, None, None),
            )
        except IntegrityError as e:
            # Check if it's a unique constraint violation
            if "UNIQUE constraint failed" in str(e):
                raise ValueError(f"Chapter with name {name} already exists") from e
            # Re-raise other integrity errors (e.g., foreign key violations)
            raise

        return Chapter(
            _cursor=cursor,
            id=chapter_id,
            name=name,
            text=text,
            conversation_id=None,
            chapter_summary_page_id=None,
        )

    @property
    def url_id(self) -> int:
        """Return the 1-based identifier used in URLs."""
        return self.id + 1

    @staticmethod
    def get_latest_started_chapter(cursor: Cursor) -> "Chapter" | None:
        """Get the latest chapter that has been started (has conversation_id set)."""
        row = cursor.execute(
            """SELECT id, name, text, conversation_id, chapter_summary_page_id
               FROM chapter
               WHERE conversation_id IS NOT NULL
               ORDER BY id DESC
               LIMIT 1"""
        ).fetchone()
        if row is None:
            return None
        return Chapter._from_row(cursor, row)

    @staticmethod
    def read_chapter(cursor: Cursor, chapter_id: int) -> "Chapter" | None:
        row = cursor.execute(
            """SELECT id, name, text, conversation_id, chapter_summary_page_id
               FROM chapter WHERE id = ?""",
            (chapter_id,),
        ).fetchone()
        if row is None:
            return None
        return Chapter._from_row(cursor, row)

    @staticmethod
    def find_first_unstarted_chapter(cursor: Cursor) -> "Chapter" | None:
        """Find the first chapter where conversation_id is NULL."""
        row = cursor.execute(
            """SELECT id, name, text, conversation_id, chapter_summary_page_id
               FROM chapter
               WHERE conversation_id IS NULL
               ORDER BY id
               LIMIT 1"""
        ).fetchone()
        if row is None:
            return None
        return Chapter._from_row(cursor, row)

    @staticmethod
    def get_chapter_count(cursor: Cursor) -> int:
        """Get the total number of chapters in the database."""
        row = cursor.execute("SELECT COUNT(*) FROM chapter").fetchone()
        return row[0] if row else 0

    @staticmethod
    def get_started_chapter_names(cursor: Cursor) -> list[ChapterName]:
        """Get the id and names of all started chapters, ordered by id."""
        rows = cursor.execute(
            """SELECT id, name FROM chapter
               WHERE conversation_id IS NOT NULL
               ORDER BY id ASC"""
        ).fetchall()
        return [ChapterName(id=row["id"], name=json.loads(row["name"])) for row in rows]

    @staticmethod
    def get_page_of_chapters(
        cursor: Cursor, offset: int = 0, count: int = 10000
    ) -> list[tuple["Chapter", int]]:
        """Get a page of chapters with their wiki page counts.

        Returns list of (Chapter, wiki_count) tuples.
        """
        rows = cursor.execute(
            """SELECT c.id, c.name, c.text, c.conversation_id,
                      c.chapter_summary_page_id, COUNT(DISTINCT wp.slug) as wiki_count
               FROM chapter c
               LEFT JOIN wiki_page wp ON wp.chapter = c.id
               GROUP BY c.id, c.name, c.text, c.conversation_id,
                        c.chapter_summary_page_id
               ORDER BY c.id
               LIMIT ? OFFSET ?""",
            (count, offset),
        ).fetchall()

        result = []
        for row in rows:
            chapter = Chapter._from_row(cursor, row)
            result.append((chapter, row["wiki_count"]))
        return result

    # TODO: deprecated
    def set_conversation_id(self, conversation: "Conversation") -> None:
        """Set the conversation_id field for this chapter."""
        logger.info(
            f"Setting chapter {self.id} as started by conversation {conversation.id}"
        )
        self._cursor.execute(
            "UPDATE chapter SET conversation_id = ? WHERE id = ?",
            (conversation.id, self.id),
        )

    def start_chapter(self, conversation: "Conversation") -> None:
        from .wikipage import WikiPage

        self.set_conversation_id(conversation)
        WikiPage.copy_current_for_new_chapter(self._cursor, self.id)

    @cached_property
    def created_pages(self) -> list["WikiPage"]:
        """Get the latest version of each wiki page created for the first time.

        Only includes pages from this chapter.

        Returns pages where the slug appears for the first time in this chapter.
        """
        from .wikipage import WikiPage

        rows = self._cursor.execute(
            """WITH created_slugs AS (
                   SELECT DISTINCT wp1.slug
                   FROM wiki_page wp1
                   WHERE wp1.chapter = ?
                   AND NOT EXISTS (
                       SELECT 1 FROM wiki_page wp2
                       WHERE wp2.slug = wp1.slug AND wp2.chapter < ?
                   )
               ),
               latest_created_pages AS (
                   SELECT wp.slug, MAX(wp.id) as max_id
                   FROM wiki_page wp
                   INNER JOIN created_slugs cs ON wp.slug = cs.slug
                   WHERE wp.chapter = ?
                   GROUP BY wp.slug
               ),
               filtered_created_pages AS (
                   SELECT lcp.slug, lcp.max_id
                   FROM latest_created_pages lcp
                   INNER JOIN wiki_page wp ON wp.id = lcp.max_id
                   WHERE wp.title != ''
               )
               SELECT wp.id, wp.slug, wp.title, wp.summary, wp.body, wp.create_time,
                      wp.chapter, wp.create_block
               FROM wiki_page wp
               INNER JOIN filtered_created_pages lcp ON wp.id = lcp.max_id
               ORDER BY wp.slug""",
            (self.id, self.id, self.id),
        ).fetchall()

        return [WikiPage._from_row(self._cursor, row) for row in rows]

    @cached_property
    def updated_pages(self) -> list["WikiPage"]:
        """Get the latest version of each wiki page updated in this chapter.

        Returns pages where the slug existed in previous chapters and was modified here.
        """
        from .wikipage import WikiPage

        rows = self._cursor.execute(
            """WITH updated_slugs AS (
                   SELECT DISTINCT wp1.slug
                   FROM wiki_page wp1
                   WHERE wp1.chapter = ?
                   AND EXISTS (
                       SELECT 1 FROM wiki_page wp2
                       WHERE wp2.slug = wp1.slug AND wp2.chapter < ?
                   )
               ),
               latest_updated_pages AS (
                   SELECT wp.slug, MAX(wp.id) as max_id
                   FROM wiki_page wp
                   INNER JOIN updated_slugs us ON wp.slug = us.slug
                   WHERE wp.chapter = ?
                   GROUP BY wp.slug
               ),
               filtered_updated_pages AS (
                   SELECT lup.slug, lup.max_id
                   FROM latest_updated_pages lup
                   INNER JOIN wiki_page wp ON wp.id = lup.max_id
                   WHERE wp.title != ''
               )
               SELECT wp.id, wp.slug, wp.title, wp.summary, wp.body, wp.create_time,
                      wp.chapter, wp.create_block
               FROM wiki_page wp
               INNER JOIN filtered_updated_pages lup ON wp.id = lup.max_id
               ORDER BY wp.slug""",
            (self.id, self.id, self.id),
        ).fetchall()

        return [WikiPage._from_row(self._cursor, row) for row in rows]

    @cached_property
    def conversation(self) -> "Conversation | None":
        """Get the conversation associated with this chapter, if any."""
        if not self.conversation_id:
            return None

        from .conversation import Conversation

        return Conversation.get_by_id(self._cursor, self.conversation_id)

    @cached_property
    def chapter_summary_page(self) -> "WikiPage | None":
        """Get the wiki page that summarizes this chapter, if any."""
        if not self.chapter_summary_page_id:
            return None

        from .wikipage import WikiPage

        return WikiPage.get_by_id(self._cursor, self.chapter_summary_page_id)

    def set_chapter_summary_page(self, wiki_page: "WikiPage") -> None:
        """Set the chapter summary page for this chapter."""
        logger.info(
            f"Setting chapter {self.id} summary page to {wiki_page.id} "
            f"(slug: {wiki_page.slug})"
        )
        self._cursor.execute(
            "UPDATE chapter SET chapter_summary_page_id = ? WHERE id = ?",
            (wiki_page.id, self.id),
        )

    @staticmethod
    def _from_row(cursor: Cursor, row: Row) -> "Chapter":
        """Create a Chapter instance from a database row."""
        return Chapter(
            _cursor=cursor,
            id=row["id"],
            name=json.loads(row["name"]),
            text=row["text"],
            conversation_id=row["conversation_id"],
            chapter_summary_page_id=row["chapter_summary_page_id"],
        )
