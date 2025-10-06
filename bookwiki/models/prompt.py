from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from functools import cached_property
from sqlite3 import Cursor, Row
from string import Template

# Import TYPE_CHECKING to avoid circular imports
from typing import TYPE_CHECKING

from bookwiki.utils import utc_now

if TYPE_CHECKING:
    from .block import Block

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Prompt:
    _cursor: Cursor
    key: str
    create_time: datetime
    create_block_id: int
    summary: str
    template: Template

    @staticmethod
    def create(
        cursor: Cursor,
        key: str,
        create_block_id: int,
        summary: str,
        template: Template,
    ) -> "Prompt":
        # Validate that key is not empty
        if not key:
            raise ValueError("Prompt key cannot be empty")

        now = utc_now()
        cursor.execute(
            """INSERT INTO prompt (key, create_time, create_block, summary, template)
               VALUES (?, ?, ?, ?, ?)""",
            (
                key,
                now.isoformat(),
                create_block_id,
                summary,
                template.template,
            ),
        )

        return Prompt(
            _cursor=cursor,
            key=key,
            create_time=now,
            create_block_id=create_block_id,
            summary=summary,
            template=template,
        )

    @staticmethod
    def list_prompts(cursor: Cursor) -> dict[str, "Prompt"]:
        rows = cursor.execute(
            """SELECT key, MAX(create_time) as latest_time
               FROM prompt GROUP BY key"""
        ).fetchall()

        prompts = {}
        for row in rows:
            prompt_row = cursor.execute(
                "SELECT * FROM prompt WHERE key = ? AND create_time = ?",
                (row["key"], row["latest_time"]),
            ).fetchone()

            prompts[row["key"]] = Prompt._from_row(cursor, prompt_row)
        return prompts

    @staticmethod
    def get_prompt(
        cursor: Cursor, key: str, create_time: datetime | None = None
    ) -> "Prompt" | None:
        if create_time is None:
            row = cursor.execute(
                """SELECT * FROM prompt WHERE key = ?
                   ORDER BY create_time DESC LIMIT 1""",
                (key,),
            ).fetchone()
        else:
            row = cursor.execute(
                """SELECT * FROM prompt WHERE key = ? AND create_time <= ?
                   ORDER BY create_time DESC LIMIT 1""",
                (key, create_time.isoformat()),
            ).fetchone()

        if row is None:
            return None

        return Prompt._from_row(cursor, row)

    @staticmethod
    def get_all_versions(cursor: Cursor, key: str) -> list["Prompt"]:
        """Get all versions of this prompt, ordered by create_time DESC."""
        rows = cursor.execute(
            """SELECT * FROM prompt WHERE key = ?
               ORDER BY create_time DESC""",
            (key,),
        ).fetchall()

        versions = []
        for row in rows:
            versions.append(Prompt._from_row(cursor, row))
        return versions

    @cached_property
    def version_count(self) -> int:
        row = self._cursor.execute(
            "SELECT COUNT(*) FROM prompt WHERE key = ?", (self.key,)
        ).fetchone()
        assert row is not None
        count: int = row[0]
        return count

    @cached_property
    def create_block(self) -> "Block":
        from .block import Block

        ret = Block.get_by_id(self._cursor, self.create_block_id)
        assert ret is not None
        return ret

    @staticmethod
    def _from_row(cursor: Cursor, row: Row) -> "Prompt":
        """Create a Prompt instance from a database row."""
        return Prompt(
            _cursor=cursor,
            key=row["key"],
            create_time=datetime.fromisoformat(row["create_time"]),
            create_block_id=row["create_block"],
            summary=row["summary"],
            template=Template(row["template"]),
        )
