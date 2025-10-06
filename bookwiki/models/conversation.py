from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from functools import cached_property
from sqlite3 import Cursor, Row

# Import TYPE_CHECKING to avoid circular imports
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .block import Block
    from .chapter import Chapter

logger = logging.getLogger(__name__)


class ConversationStatus(Enum):
    """Status of conversation processing."""

    WAITING_LLM = "Waiting LLM"
    WAITING_TOOLS = "Waiting Tools"
    UNSENT = "Ready"
    FINISHED = "Finished"


@dataclass(frozen=True)
class ConversationStats:
    """Statistics for a conversation."""

    block_count: int
    last_activity: Optional[datetime]
    sent_blocks: int
    unsent_blocks: int
    blocks_ready_to_send: int
    blocks_waiting_responses: int
    unique_tools: int


@dataclass(frozen=True)
class ConversationsStats:
    """Aggregated statistics for all conversations."""

    total_input_tokens: int
    total_output_tokens: int
    conversations_waiting_llm: int
    conversations_waiting_tools: int
    conversations_ready: int
    conversations_finished: int


@dataclass(frozen=True)
class Conversation:
    _cursor: Cursor
    id: int
    previously: Optional[str]  # Opaque string for LLM service
    parent_block_id: Optional[int]
    total_input_tokens: int
    total_output_tokens: int
    current_tokens: int
    current_generation: int
    waiting_on_id: Optional[str]

    @staticmethod
    def get_by_id(cursor: Cursor, conversation_id: int) -> Optional["Conversation"]:
        """Get a conversation by its ID.

        Args:
            cursor: Database cursor
            conversation_id: The ID of the conversation to retrieve

        Returns:
            Conversation object if found, None otherwise
        """
        row = cursor.execute(
            "SELECT * FROM conversation WHERE id = ?", (conversation_id,)
        ).fetchone()

        if row is None:
            return None

        return Conversation._from_row(cursor, row)

    @staticmethod
    def get_by_parent_block_id(
        cursor: Cursor, parent_block_id: int
    ) -> Optional["Conversation"]:
        """Get a conversation by its parent block ID.

        Args:
            cursor: Database cursor
            parent_block_id: The ID of the parent block

        Returns:
            Conversation object if found, None otherwise
        """
        row = cursor.execute(
            "SELECT * FROM conversation WHERE parent_block = ?", (parent_block_id,)
        ).fetchone()

        if row is None:
            return None

        return Conversation._from_row(cursor, row)

    @staticmethod
    def create(cursor: Cursor, parent_block_id: Optional[int] = None) -> "Conversation":
        cursor.execute(
            """INSERT INTO conversation (previously, parent_block, total_input_tokens,
               total_output_tokens, current_tokens, current_generation, waiting_on_id)
               VALUES (?, ?, 0, 0, 0, 0, ?)""",
            (None, parent_block_id, None),
        )
        conv_id = cursor.lastrowid
        assert conv_id is not None
        logger.info(
            f"Created new conversation {conv_id}"
            + (f" with parent block {parent_block_id}" if parent_block_id else "")
        )
        return Conversation(
            _cursor=cursor,
            id=conv_id,
            previously=None,
            parent_block_id=parent_block_id,
            total_input_tokens=0,
            total_output_tokens=0,
            current_tokens=0,
            current_generation=0,
            waiting_on_id=None,
        )

    @staticmethod
    def find_sendable_conversation(cursor: Cursor) -> "Conversation" | None:
        # A conversation is sendable when it has unsent messages and all tool uses have
        # responses, and it's not waiting on a result from the llm.
        # This means:
        # 1. Has at least one unsent block
        # 2. All tool uses in unsent blocks have responses (tool_response IS NOT NULL)
        # 3. waiting_on_id is NULL
        row = cursor.execute(
            """SELECT c.* FROM conversation c
               WHERE (
                   -- Has unsent blocks
                   EXISTS (
                       SELECT 1 FROM block b
                       WHERE b.conversation = c.id AND b.sent = 0
                   )
                   -- And no incomplete tool uses (tool use without response)
                   AND NOT EXISTS (
                       SELECT 1 FROM block b
                       WHERE b.conversation = c.id
                       AND b.sent = 0
                       AND b.tool_name IS NOT NULL
                       AND b.tool_response IS NULL
                   )
                   -- And not waiting on a result from the llm
                   AND c.waiting_on_id IS NULL
               )
               ORDER BY c.id
               LIMIT 1"""
        ).fetchone()

        if row is None:
            return None

        return Conversation._from_row(cursor, row)

    @staticmethod
    def find_waiting_conversation(
        cursor: Cursor, after_id: int | None = None
    ) -> "Conversation" | None:
        """Find the next conversation that has non-null waiting_on_id.

        Args:
            after_id: If provided, returns the next waiting conversation with
                ID > after_id. This allows iteration through all waiting conversations.
        """
        if after_id is None:
            row = cursor.execute(
                """SELECT * FROM conversation
                   WHERE waiting_on_id IS NOT NULL
                   ORDER BY id
                   LIMIT 1"""
            ).fetchone()
        else:
            row = cursor.execute(
                """SELECT * FROM conversation
                   WHERE waiting_on_id IS NOT NULL AND id > ?
                   ORDER BY id
                   LIMIT 1""",
                (after_id,),
            ).fetchone()

        if row is None:
            return None

        return Conversation._from_row(cursor, row)

    @staticmethod
    def all_conversations_finished(cursor: Cursor) -> bool:
        """Check if all conversations are finished.

        A conversation is considered active if it has unsent blocks or is
        waiting on a response from the LLM service.

        Returns True if there are no active conversations.
        """
        row = cursor.execute(
            """SELECT COUNT(*) FROM conversation c
               WHERE (
                   EXISTS (
                       SELECT 1 FROM block b
                       WHERE b.conversation = c.id AND b.sent = 0
                   )
                   OR c.waiting_on_id IS NOT NULL
               )"""
        ).fetchone()
        assert row is not None
        active_count: int = row[0]

        return active_count == 0

    @staticmethod
    def get_root_conversations(cursor: Cursor) -> list["Conversation"]:
        """Get all root conversations (those without a parent block).

        Returns:
            List of Conversation objects that have no parent block.
        """
        rows = cursor.execute(
            """
            SELECT * FROM conversation
            WHERE parent_block IS NULL
            ORDER BY id DESC
            """
        ).fetchall()

        conversations = []
        for row in rows:
            conversations.append(Conversation._from_row(cursor, row))
        return conversations

    @staticmethod
    def get_finished_conversations_token_percentiles(
        cursor: Cursor,
    ) -> list[tuple[str, int]]:
        """Get token percentiles for finished conversations.

        Returns:
            List of (percentile_label, token_count) tuples.
        """
        # Get all finished conversations' token counts
        rows = cursor.execute(
            """
            SELECT current_tokens
            FROM conversation
            WHERE NOT EXISTS (
                SELECT 1 FROM block
                WHERE block.conversation = conversation.id AND block.sent = 0
            )
            AND waiting_on_id IS NULL
            ORDER BY current_tokens DESC
            """
        ).fetchall()

        if not rows:
            # No finished conversations
            return [("MAX", 0), ("MIN", 0)]

        token_counts = [row[0] for row in rows]
        count = len(token_counts)

        def get_percentile_index(percentile: float) -> int:
            """Calculate the index for a given percentile."""
            # Using ceiling for percentile calculation
            index = int((count - 1) * (1 - percentile))
            return min(max(0, index), count - 1)

        return [
            ("MAX", token_counts[0]),
            ("99%", token_counts[get_percentile_index(0.99)]),
            ("90%", token_counts[get_percentile_index(0.90)]),
            ("75%", token_counts[get_percentile_index(0.75)]),
            ("50%", token_counts[get_percentile_index(0.50)]),
            ("25%", token_counts[get_percentile_index(0.25)]),
            ("MIN", token_counts[-1]),
        ]

    @staticmethod
    def get_all_conversations_stats(cursor: Cursor) -> ConversationsStats:
        """Get aggregated statistics for all conversations.

        Returns:
            ConversationsStats object with total input and output tokens across all
            conversations, plus counts by conversation status.
        """
        # Get token totals
        row = cursor.execute(
            """
            SELECT COALESCE(SUM(total_input_tokens), 0) as total_input,
                   COALESCE(SUM(total_output_tokens), 0) as total_output
            FROM conversation
            """
        ).fetchone()

        # Get all conversations and count by status
        conversations = cursor.execute("SELECT * FROM conversation").fetchall()
        status_counts = {
            ConversationStatus.WAITING_LLM: 0,
            ConversationStatus.WAITING_TOOLS: 0,
            ConversationStatus.UNSENT: 0,
            ConversationStatus.FINISHED: 0,
        }

        for conv_row in conversations:
            conv = Conversation._from_row(cursor, conv_row)
            status_counts[conv.status] += 1

        return ConversationsStats(
            total_input_tokens=row[0],
            total_output_tokens=row[1],
            conversations_waiting_llm=status_counts[ConversationStatus.WAITING_LLM],
            conversations_waiting_tools=status_counts[ConversationStatus.WAITING_TOOLS],
            conversations_ready=status_counts[ConversationStatus.UNSENT],
            conversations_finished=status_counts[ConversationStatus.FINISHED],
        )

    def update_tokens(self, input_tokens: int, output_tokens: int) -> None:
        self._cursor.execute(
            """UPDATE conversation
               SET total_input_tokens = total_input_tokens + ?,
                   total_output_tokens = total_output_tokens + ?,
                   current_tokens = ?
               WHERE id = ?""",
            (input_tokens, output_tokens, input_tokens + output_tokens, self.id),
        )

    def set_waiting_on_id(self, value: str | None) -> None:
        """Set or clear the waiting_on_id for this conversation."""
        if value:
            logger.debug(f"Conversation {self.id} now waiting on response {value}")
        else:
            logger.debug(f"Conversation {self.id} cleared waiting status")
        self._cursor.execute(
            "UPDATE conversation SET waiting_on_id = ? WHERE id = ?",
            (value, self.id),
        )

    def update_previously(self, value: str) -> None:
        """Update the previously field with the latest response ID from the LLM."""
        logger.debug(f"Conversation {self.id} updating previously to {value}")
        self._cursor.execute(
            "UPDATE conversation SET previously = ? WHERE id = ?",
            (value, self.id),
        )
        # Note: We don't update self.previously here since Conversation instances
        # are typically short-lived within a transaction context

    def increment_generation(self) -> "Conversation":
        """Increment the current_generation by one."""
        logger.debug(f"Conversation {self.id} incrementing generation")
        self._cursor.execute(
            """UPDATE conversation SET current_generation = current_generation + 1
               WHERE id = ?""",
            (self.id,),
        )
        updated = Conversation.get_by_id(self._cursor, self.id)
        assert updated is not None, f"Conversation {self.id} disappeared after update"
        return updated

    def mark_all_blocks_as_sent(self) -> None:
        """Mark all unsent blocks in this conversation as sent."""
        self._cursor.execute(
            "UPDATE block SET sent = 1 WHERE conversation = ? AND sent = 0", (self.id,)
        )

    def add_tool_use(self, name: str, use_id: str, params: str) -> "Block":
        from .block import Block

        logger.debug(f"Adding tool use '{name}' to conversation {self.id}")
        return Block.create_tool_use(
            cursor=self._cursor,
            conversation_id=self.id,
            generation=self.current_generation,
            name=name,
            use_id=use_id,
            params=params,
        )

    def add_assistant_text(self, text: str) -> "Block":
        # Assistant text is marked as sent=True since it's from the LLM
        from .block import Block

        return Block.create_text(
            cursor=self._cursor,
            conversation_id=self.id,
            generation=self.current_generation,
            role="assistant",
            text=text,
            sent=True,
        )

    def add_user_text(self, text: str) -> "Block":
        from .block import Block

        return Block.create_text(
            cursor=self._cursor,
            conversation_id=self.id,
            generation=self.current_generation,
            role="user",
            text=text,
            sent=False,
        )

    @cached_property
    def blocks(self) -> list["Block"]:
        from .block import Block

        rows = self._cursor.execute(
            "SELECT * FROM block WHERE conversation = ? ORDER BY id", (self.id,)
        ).fetchall()

        blocks = []
        for row in rows:
            blocks.append(Block._from_row(self._cursor, row))
        return blocks

    @cached_property
    def unsent_blocks(self) -> list["Block"]:
        """Get all blocks that haven't been sent yet."""
        from .block import Block

        rows = self._cursor.execute(
            "SELECT * FROM block WHERE conversation = ? AND sent = 0 ORDER BY id",
            (self.id,),
        ).fetchall()

        return [Block._from_row(self._cursor, row) for row in rows]

    @cached_property
    def parent_block(self) -> "Block" | None:
        """Get the parent block for this conversation, if it exists."""
        if self.parent_block_id is None:
            return None
        from .block import Block

        return Block.get_by_id(self._cursor, self.parent_block_id)

    @cached_property
    def chapter(self) -> "Chapter":
        """Get the chapter associated with this conversation.

        If this conversation started a chapter, returns that chapter.
        Otherwise, follows the parent chain until finding a conversation
        that started a chapter.

        Returns:
            Chapter object associated with this conversation or its ancestors

        Raises:
            ValueError: If no chapter can be found (violates system invariants)
        """
        from .chapter import Chapter

        # First check if this conversation directly started a chapter
        row = self._cursor.execute(
            "SELECT * FROM chapter WHERE conversation_id = ?", (self.id,)
        ).fetchone()

        if row is not None:
            return Chapter._from_row(self._cursor, row)

        # If not, follow the parent chain
        if self.parent_block_id is not None:
            parent_block = self.parent_block
            if parent_block is not None:
                # Get the conversation that owns this parent block
                parent_conv = Conversation.get_by_id(
                    self._cursor, parent_block.conversation_id
                )
                if parent_conv is not None:
                    return parent_conv.chapter

        # If we reach here, the system invariants have been violated
        raise ValueError(
            f"Conversation {self.id} has no associated chapter. "
            "Every conversation must either directly start a chapter "
            "or have a parent conversation."
        )

    @cached_property
    def stats(self) -> ConversationStats:
        """Get statistics for this conversation.

        Returns:
            ConversationStats object with comprehensive block statistics.
        """
        row = self._cursor.execute(
            """
            SELECT COUNT(*) as block_count,
                   MAX(create_time) as last_activity,
                   SUM(CASE WHEN sent = 1 THEN 1 ELSE 0 END) as sent_blocks,
                   SUM(CASE WHEN sent = 0 THEN 1 ELSE 0 END) as unsent_blocks,
                   SUM(CASE WHEN sent = 0 AND
                       (tool_name IS NULL OR tool_response IS NOT NULL)
                       THEN 1 ELSE 0 END) as blocks_ready_to_send,
                   SUM(CASE WHEN sent = 0 AND tool_name IS NOT NULL AND
                       tool_response IS NULL
                       THEN 1 ELSE 0 END) as blocks_waiting_responses,
                   COUNT(DISTINCT tool_name) as unique_tools
            FROM block
            WHERE conversation = ?
            """,
            (self.id,),
        ).fetchone()

        last_activity = None
        if row["last_activity"]:
            last_activity = datetime.fromisoformat(row["last_activity"])

        return ConversationStats(
            block_count=row["block_count"] or 0,
            last_activity=last_activity,
            sent_blocks=row["sent_blocks"] or 0,
            unsent_blocks=row["unsent_blocks"] or 0,
            blocks_ready_to_send=row["blocks_ready_to_send"] or 0,
            blocks_waiting_responses=row["blocks_waiting_responses"] or 0,
            unique_tools=row["unique_tools"] or 0,
        )

    @cached_property
    def children(self) -> list["Conversation"]:
        """Get all child conversations (those with parent blocks in this conversation).

        Returns:
            List of Conversation objects that were spawned from blocks in this
            conversation.
        """
        rows = self._cursor.execute(
            """
            SELECT c.* FROM conversation c
            WHERE c.parent_block IN (
                SELECT id FROM block WHERE conversation = ?
            )
            ORDER BY c.id ASC
            """,
            (self.id,),
        ).fetchall()

        conversations = []
        for row in rows:
            conversations.append(Conversation._from_row(self._cursor, row))
        return conversations

    @cached_property
    def status(self) -> ConversationStatus:
        """Determine the current status of this conversation.

        Returns:
            ConversationStatus indicating the current state of processing.
        """
        # Check if waiting on LLM response
        if self.waiting_on_id is not None:
            return ConversationStatus.WAITING_LLM

        # Check for unsent blocks
        has_unsent = (
            self._cursor.execute(
                "SELECT 1 FROM block WHERE conversation = ? AND sent = 0 LIMIT 1",
                (self.id,),
            ).fetchone()
            is not None
        )

        if not has_unsent:
            return ConversationStatus.FINISHED

        # Check if any unsent tool uses are waiting for responses
        waiting_tools = (
            self._cursor.execute(
                """SELECT 1 FROM block
               WHERE conversation = ?
               AND sent = 0
               AND tool_name IS NOT NULL
               AND tool_response IS NULL
               LIMIT 1""",
                (self.id,),
            ).fetchone()
            is not None
        )

        if waiting_tools:
            return ConversationStatus.WAITING_TOOLS

        # Has unsent blocks ready to send
        return ConversationStatus.UNSENT

    def detect_serial_tool_use(self) -> bool:
        """Detect if any tool type was used serially in the last two generations.

        Returns True if there's exactly one tool use in each of the last two
        generations and they're the same tool type.
        """
        # Get tool usage summary for last two generations
        rows = self._cursor.execute(
            """SELECT tool_name, generation, COUNT(*) as count
               FROM block
               WHERE conversation = ?
               AND generation >= ?
               AND tool_name IS NOT NULL
               GROUP BY tool_name, generation""",
            (self.id, self.current_generation - 1),
        ).fetchall()

        # Serial usage: exactly 2 rows, different generations, count=1 for each
        return (
            len(rows) == 2
            and rows[0]["count"] == 1
            and rows[1]["count"] == 1
            and rows[0]["tool_name"] == rows[1]["tool_name"]
            and rows[0]["generation"] != rows[1]["generation"]
        )

    @staticmethod
    def _from_row(cursor: Cursor, row: Row) -> "Conversation":
        """Create a Conversation instance from a database row."""
        return Conversation(
            _cursor=cursor,
            id=row["id"],
            previously=row["previously"],
            parent_block_id=row["parent_block"],
            total_input_tokens=row["total_input_tokens"],
            total_output_tokens=row["total_output_tokens"],
            current_tokens=row["current_tokens"],
            current_generation=row["current_generation"],
            waiting_on_id=row["waiting_on_id"],
        )
