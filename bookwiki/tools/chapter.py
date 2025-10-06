from typing import Optional

from bookwiki.models import Block, Chapter
from bookwiki.tools.base import LLMSolvableError, ToolModel


class ReadChapter(ToolModel):
    """Retrieve the full text content of a book chapter to analyze for wiki updates.

    This is your primary source of new information. Read chapters to discover
    characters, locations, events, and concepts that need wiki pages or updates.
    You can only access chapters that have been started - you cannot read ahead to
    future chapters."""

    chapter_offset: Optional[int] = None
    """Offset from the latest started chapter. If not provided, read the latest
    chapter. Negative values read earlier chapters (e.g., -1 reads the previous
    chapter). Positive values are not allowed as they would read future chapters."""

    def _apply(
        self,
        block: Block,
    ) -> None:
        # Treat None as 0
        offset = self.chapter_offset if self.chapter_offset is not None else 0

        # Validate that offset is not positive (no reading the future)
        if offset > 0:
            raise LLMSolvableError(
                "Cannot read future chapters (chapter_offset must be 0 or negative)"
            )

        # Get the latest started chapter
        latest_started = Chapter.get_latest_started_chapter(block.get_cursor())
        if latest_started is None:
            raise LLMSolvableError("No chapters have been started yet")

        # Calculate the target chapter ID based on offset
        target_chapter_id = latest_started.id + offset

        # Read the target chapter
        chapter = Chapter.read_chapter(block.get_cursor(), target_chapter_id)
        if chapter is None:
            raise LLMSolvableError(f"Chapter {target_chapter_id} does not exist")

        # Verify the chapter has been started
        if chapter.conversation_id is None:
            raise LLMSolvableError(
                f"Chapter {target_chapter_id} has not been started yet"
            )

        text = f"**{' > '.join(chapter.name)}**\n\n{chapter.text}"
        block.respond(text)
