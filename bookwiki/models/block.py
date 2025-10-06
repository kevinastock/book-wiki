from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from functools import cached_property
from sqlite3 import Cursor, Row
from string import Template

# Import TYPE_CHECKING to avoid circular imports
from typing import TYPE_CHECKING, Any, Optional

from bookwiki.utils import utc_now

if TYPE_CHECKING:
    from .conversation import Conversation
    from .prompt import Prompt
    from .wikipage import WikiPage

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PaginatedToolUse:
    """Paginated tool use results."""

    blocks: list["Block"]
    total_count: int
    total_pages: int


@dataclass(frozen=True)
class ToolUsageStats:
    """Tool usage statistics."""

    name: str
    used: int
    failed: int


@dataclass(frozen=True)
class Block:
    _cursor: Cursor
    id: int
    conversation_id: int
    create_time: datetime
    generation: int
    tool_name: Optional[str]
    tool_use_id: Optional[str]
    tool_params: Optional[str]
    tool_response: Optional[str]
    text_role: Optional[str]
    text_body: Optional[str]
    sent: bool
    errored: bool

    def get_cursor(self) -> Cursor:
        return self._cursor

    @staticmethod
    def get_by_id(cursor: Cursor, block_id: int) -> "Block" | None:
        """Get a Block by its ID."""
        row = cursor.execute("SELECT * FROM block WHERE id = ?", (block_id,)).fetchone()

        if row is None:
            return None

        return Block._from_row(cursor, row)

    @staticmethod
    def get_unresponded_blocks(cursor: Cursor, tool_name: str) -> list["Block"]:
        """Get all blocks with the given tool_name that have no response."""
        rows = cursor.execute(
            """
            SELECT * FROM block
            WHERE tool_name = ?
            AND (tool_response IS NULL OR tool_response = '')
            ORDER BY create_time DESC
            """,
            (tool_name,),
        ).fetchall()

        blocks = []
        for row in rows:
            blocks.append(Block._from_row(cursor, row))
        return blocks

    @staticmethod
    def create_tool_use(
        cursor: Cursor,
        conversation_id: int,
        generation: int,
        name: str,
        use_id: str,
        params: str,
    ) -> "Block":
        """Create a new tool use block."""
        now = utc_now()
        cursor.execute(
            """INSERT INTO block (conversation, create_time, generation, tool_name,
               tool_use_id, tool_params, sent, errored)
               VALUES (?, ?, ?, ?, ?, ?, 0, 0)""",
            (conversation_id, now.isoformat(), generation, name, use_id, params),
        )

        block_id = cursor.lastrowid
        assert block_id is not None
        return Block(
            _cursor=cursor,
            id=block_id,
            conversation_id=conversation_id,
            create_time=now,
            generation=generation,
            tool_name=name,
            tool_use_id=use_id,
            tool_params=params,
            tool_response=None,
            text_role=None,
            text_body=None,
            sent=False,
            errored=False,
        )

    @staticmethod
    def create_text(
        cursor: Cursor,
        conversation_id: int,
        generation: int,
        role: str,
        text: str,
        sent: bool = False,
    ) -> "Block":
        """Create a new text block (user or assistant message)."""
        now = utc_now()
        cursor.execute(
            """INSERT INTO block (conversation, create_time, generation, text_role,
               text_body, sent, errored) VALUES (?, ?, ?, ?, ?, ?, 0)""",
            (conversation_id, now.isoformat(), generation, role, text, sent),
        )

        block_id = cursor.lastrowid
        assert block_id is not None
        return Block(
            _cursor=cursor,
            id=block_id,
            conversation_id=conversation_id,
            create_time=now,
            generation=generation,
            tool_name=None,
            tool_use_id=None,
            tool_params=None,
            tool_response=None,
            text_role=role,
            text_body=text,
            sent=sent,
            errored=False,
        )

    def mark_as_sent(self) -> None:
        self._cursor.execute("UPDATE block SET sent = 1 WHERE id = ?", (self.id,))

    def respond(self, response: str) -> None:
        # Check current state in database
        row = self._cursor.execute(
            "SELECT tool_response FROM block WHERE id = ?", (self.id,)
        ).fetchone()
        if row is not None and row["tool_response"] is not None:
            raise ValueError(
                f"Cannot respond to block {self.id}: tool already has a response"
            )
        self._cursor.execute(
            "UPDATE block SET tool_response = ? WHERE id = ?", (response, self.id)
        )

    def respond_error(self, message: str) -> None:
        # Check current state in database
        row = self._cursor.execute(
            "SELECT tool_response FROM block WHERE id = ?", (self.id,)
        ).fetchone()
        if row is not None and row["tool_response"] is not None:
            raise ValueError(
                f"Cannot respond to block {self.id}: tool already has a response"
            )
        self._cursor.execute(
            "UPDATE block SET tool_response = ?, errored = 1 WHERE id = ?",
            (message, self.id),
        )

    def add_prompt(self, key: str, summary: str, template: Template) -> "Prompt":
        from .prompt import Prompt

        return Prompt.create(
            cursor=self._cursor,
            key=key,
            create_block_id=self.id,
            summary=summary,
            template=template,
        )

    def write_wiki_page(
        self,
        chapter_id: int,
        slug: str,
        title: str,
        names: list[str],
        summary: str,
        body: str,
    ) -> "WikiPage":
        from .wikipage import WikiPage

        return WikiPage.create(
            cursor=self._cursor,
            chapter_id=chapter_id,
            slug=slug,
            create_block_id=self.id,
            title=title,
            names=names,
            summary=summary,
            body=body,
        )

    def start_conversation(self) -> "Conversation":
        # Verify this is a tool use block without a response
        if self.tool_name is None or self.tool_use_id is None:
            raise ValueError("Can only start conversations from tool use blocks")
        if self.tool_response is not None:
            raise ValueError(
                "Cannot start conversation from a tool use that already has a response"
            )
        from .conversation import Conversation

        return Conversation.create(self._cursor, parent_block_id=self.id)

    @cached_property
    def conversation(self) -> "Conversation":
        """Get the conversation this block belongs to."""
        from .conversation import Conversation

        conversation = Conversation.get_by_id(self._cursor, self.conversation_id)
        if conversation is None:
            raise ValueError(
                f"Block {self.id} references non-existent conversation "
                f"{self.conversation_id}"
            )
        return conversation

    @cached_property
    def spawned_conversation(self) -> "Conversation | None":
        """Get the conversation spawned by this block, if any."""
        from .conversation import Conversation

        return Conversation.get_by_parent_block_id(self._cursor, self.id)

    @cached_property
    def created_wiki_page(self) -> "WikiPage | None":
        """Get the wiki page created by this block, if any."""
        from .wikipage import WikiPage

        return WikiPage.get_by_create_block_id(self._cursor, self.id)

    @cached_property
    def tool_params_json(self) -> dict[str, Any]:
        return json.loads(self.tool_params) if self.tool_params else {}

    @staticmethod
    def get_blocks_by_tool_paginated(
        cursor: Cursor, tool_name: str, page: int = 1, page_size: int = 20
    ) -> PaginatedToolUse:
        """Get paginated blocks for a specific tool.

        Args:
            cursor: Database cursor
            tool_name: Name of the tool to filter by
            page: Page number (1-based)
            page_size: Number of items per page

        Returns:
            Tuple of (blocks, total_count, total_pages)
        """
        # Get total count
        total_count = cursor.execute(
            "SELECT COUNT(*) as count FROM block WHERE tool_name = ?",
            (tool_name,),
        ).fetchone()["count"]

        # Calculate pagination
        total_pages = (
            (total_count + page_size - 1) // page_size if total_count > 0 else 0
        )
        offset = (page - 1) * page_size

        # Get paginated blocks
        rows = cursor.execute(
            """
            SELECT * FROM block
            WHERE tool_name = ?
            ORDER BY create_time DESC
            LIMIT ? OFFSET ?
            """,
            (tool_name, page_size, offset),
        ).fetchall()

        blocks = [Block._from_row(cursor, row) for row in rows]
        return PaginatedToolUse(
            blocks=blocks, total_count=total_count, total_pages=total_pages
        )

    @staticmethod
    def get_tool_usage_stats(cursor: Cursor) -> list[ToolUsageStats]:
        """Get usage statistics for all tools.

        Returns:
            List of ToolUsageStats objects
        """
        rows = cursor.execute(
            """
            SELECT
                tool_name,
                COUNT(*) as total_uses,
                SUM(CASE WHEN tool_response IS NOT NULL THEN 1 ELSE 0 END) as used,
                SUM(CASE WHEN errored = 1 THEN 1 ELSE 0 END) as failed_count
            FROM block
            WHERE tool_name IS NOT NULL
            GROUP BY tool_name
            ORDER BY tool_name
            """
        ).fetchall()

        stats = []
        for row in rows:
            stats.append(
                ToolUsageStats(
                    name=row["tool_name"],
                    used=int(row["used"]),
                    failed=int(row["failed_count"]),
                )
            )
        return stats

    @staticmethod
    def _from_row(cursor: Cursor, row: Row) -> "Block":
        """Create a Block instance from a database row."""
        return Block(
            _cursor=cursor,
            id=row["id"],
            conversation_id=row["conversation"],
            create_time=datetime.fromisoformat(row["create_time"]),
            generation=row["generation"],
            tool_name=row["tool_name"],
            tool_use_id=row["tool_use_id"],
            tool_params=row["tool_params"],
            tool_response=row["tool_response"],
            text_role=row["text_role"],
            text_body=row["text_body"],
            sent=bool(row["sent"]),
            errored=bool(row["errored"]),
        )
