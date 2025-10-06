"""Processing logic for conversations."""

import logging
from sqlite3 import Cursor

from bookwiki.db import SafeConnection
from bookwiki.llm import LLMRetryableError, LLMService
from bookwiki.models import Block, Chapter, Conversation, WikiPage
from bookwiki.models.configuration import Configuration

logger = logging.getLogger(__name__)


class Processor:
    """Main processor for handling conversation loops and chapter advancement."""

    def __init__(self, connection: SafeConnection, llm_service: LLMService):
        self.connection = connection
        self.llm_service = llm_service

    def process_sendable_conversations(self) -> None:
        """Process conversations that are ready to send to LLM.

        Returns True if any conversations were sent.
        """
        while True:
            with self.connection.transaction_cursor(threshold_ms=60000.0) as cursor:
                conv = Conversation.find_sendable_conversation(cursor)
                if conv is None:
                    return

                self._send_conversation(cursor, conv)

    def process_waiting_conversations(self) -> None:
        """Process conversations waiting for LLM responses.

        Returns True if any response retrieves were attempted.
        """
        last_id = None

        # Iterate through all waiting conversations
        while True:
            with self.connection.transaction_cursor(threshold_ms=60000.0) as cursor:
                conv = Conversation.find_waiting_conversation(cursor, after_id=last_id)
                if conv is None:
                    break

                last_id = conv.id
                self._retrieve_and_handle_conversation(conv)

    def _send_conversation(self, cursor: Cursor, conv: Conversation) -> None:
        """Send a conversation to the LLM service."""
        compressing = False
        system_message = ""  # Use default
        if conv.current_tokens > self.llm_service.get_compression_threshold():
            compressing = True
            logger.info(
                f"Conversation {conv.id} will be compressed "
                f"(tokens: {conv.current_tokens})"
            )
            compress_prompt = Configuration.get_compress_prompt(cursor)
            conv.add_user_text(compress_prompt)
            system_message = "You are an intelligent agent helping another agent."

        # Log what we're sending for debugging
        unsent_blocks = conv.unsent_blocks
        logger.debug(
            f"Sending conversation {conv.id} with {len(unsent_blocks)} unsent blocks, "
            f"previously={conv.previously}"
        )

        response_id = self.llm_service.prompt(
            previously=conv.previously,
            new_messages=unsent_blocks,
            system_message=system_message,
            compressing=compressing,
        )

        logger.info(f"Conversation {conv.id} sent with response_id {response_id}")
        conv.set_waiting_on_id(response_id)

    def _retrieve_and_handle_conversation(self, conv: Conversation) -> None:
        """Retrieve and handle a waiting conversation response."""
        assert conv.waiting_on_id is not None

        logger.debug(
            f"Retrieving response for conversation {conv.id} "
            f"(response_id: {conv.waiting_on_id})"
        )

        try:
            response = self.llm_service.try_fetch(conv.waiting_on_id)
        except LLMRetryableError as e:
            # Clear the response ID so it will be resubmitted on next iteration
            logger.warning(
                f"Retryable error for conversation {conv.id}: {e}. Will retry."
            )
            conv.set_waiting_on_id(None)
            return

        if response is None:
            logger.debug(f"No response ready yet for conversation {conv.id}")
            return

        logger.info(
            f"Received response for conversation {conv.id} "
            f"(input_tokens: {response.input_tokens}, "
            f"output_tokens: {response.output_tokens})"
        )

        # Update the generation so blocks from this response have a new value.
        conv = conv.increment_generation()

        conv.set_waiting_on_id(None)
        conv.mark_all_blocks_as_sent()
        conv.update_tokens(response.input_tokens, response.output_tokens)

        # Update the previously field with the new response ID for future requests
        conv.update_previously(response.updated_prev)
        logger.debug(
            f"Updated conversation {conv.id} previously to {response.updated_prev}"
        )

        if response.compressing:
            for text in response.texts:
                # The assistant response is actually treated as the new text sent to
                # the LLM, so it's user text.
                conv.add_user_text(text)
            return

        last_assistant_block = None
        for text in response.texts:
            last_assistant_block = conv.add_assistant_text(text)

        for tool_model in response.tools:
            # Use the new add_to_conversation method from ToolModel
            tool_block = tool_model.add_to_conversation(conv)
            logger.debug(
                f"Applying tool {tool_model.__class__.__name__} "
                f"for conversation {conv.id}"
            )
            tool_model.apply(tool_block)

        # Check if we should suggest parallel tool execution
        if conv.detect_serial_tool_use():
            logger.info(f"Detected serial tool use in conversation {conv.id}.")

            suggestion = (
                "For better performance, consider using multiple tool calls in "
                "parallel if they don't depend on each other."
            )
            conv.add_user_text(suggestion)

        if not response.tools:
            # If there are no tool uses and this is a sub-conversation,
            # respond to the parent block with the combined response texts
            parent_block = conv.parent_block
            if parent_block is not None:
                # Combine response texts with double newlines
                combined_response = "\n\n".join(response.texts)
                parent_block.respond(combined_response)
                logger.info(
                    f"Responded to parent block {parent_block.id} "
                    f"with combined text response"
                )
            else:
                # Finalize the chapter when there are no tool uses
                # This ensures proper chapter summary handling
                self._finalize_chapter(conv, last_assistant_block)

        return

    def _finalize_chapter(
        self, conv: Conversation, last_assistant_block: Block | None
    ) -> None:
        """Finalize chapter by ensuring chapter summary exists and is properly handled.

        Args:
            conv: The conversation that has completed
            last_assistant_block: The last assistant text block from the response,
                                 used for wiki page deletion/redirect operations

        This method:
        1. Ensures a 'chapter-summary' slug exists for the chapter
        2. Updates the Chapter model to point to that summary page
        3. Deletes and redirects the chapter summary page using the provided block
        """
        # Get the current chapter
        latest_chapter = Chapter.get_latest_started_chapter(conv._cursor)
        if latest_chapter is None:
            logger.warning("No started chapter found for finalization")
            return

        logger.info(f"Finalizing chapter {latest_chapter.id}")

        # Check if chapter-summary slug exists
        summary_page = WikiPage.read_page_at(
            conv._cursor, "chapter-summary", latest_chapter.id
        )

        if summary_page is None:
            # Need to create chapter summary - instruct the LLM
            logger.info(f"Chapter {latest_chapter.id} needs chapter-summary page")
            conv.add_user_text(
                "You must create a wiki page with the slug 'chapter-summary' that "
                "summarizes the key events, characters, and plot developments from "
                "this chapter."
            )
            return  # Don't finalize yet, wait for LLM to create the summary

        # Update the chapter to point to this summary page
        latest_chapter.set_chapter_summary_page(summary_page)
        logger.info(
            f"Chapter {latest_chapter.id} linked to summary page {summary_page.id}"
        )

        # Validate that we have an assistant block for deletion operations
        if last_assistant_block is None:
            raise ValueError(
                "Cannot finalize chapter: no assistant block available for "
                "deletion operations"
            )

        # Delete the chapter summary page and redirect appropriately
        # Empty redirect means remove links entirely
        updated_pages, response_message = summary_page.delete_and_redirect(
            last_assistant_block, ""
        )

        logger.info(f"Chapter {latest_chapter.id} finalized: {response_message}")
        if updated_pages:
            logger.info(
                f"Updated {len(updated_pages)} pages during chapter finalization"
            )

    def advance_chapter_if_needed(self) -> bool:
        """Advance to next chapter if current one is complete.

        Returns True if there is still work to do, False if all processing is complete.
        """
        with self.connection.transaction_cursor() as cursor:
            if not Conversation.all_conversations_finished(cursor):
                return True

            next_chapter = Chapter.find_first_unstarted_chapter(cursor)
            if next_chapter is None:
                logger.info("No more chapters to process")
                return False

            # Start new conversation for next chapter
            logger.info(f"Advancing to chapter {next_chapter.id}")
            new_conv = Conversation.create(cursor)
            next_chapter.start_chapter(new_conv)

            # Add initial user message for the chapter
            initial_prompt = Configuration.get_chapter_prompt(cursor)
            new_conv.add_user_text(initial_prompt)
            logger.info(
                f"Created new conversation {new_conv.id} for chapter {next_chapter.id}"
            )

            return True
