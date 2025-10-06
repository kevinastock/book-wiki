"""Demo script showing LLM API usage with tool calls using v2 architecture."""

import logging
import time

from bookwiki.config_enums import (
    OpenAIModel,
    OpenAIReasoningEffort,
    OpenAIServiceTier,
    OpenAIVerbosity,
)
from bookwiki.db import connect_db
from bookwiki.impls.openai import OpenAILLMService
from bookwiki.models import Block, Conversation
from bookwiki.tools.base import ToolModel

logger = logging.getLogger(__name__)


# Define the FooBar tool locally for demo purposes
class FooBar(ToolModel):
    _description = "A tool for testing, you should use it."

    colors: list[str]
    """A list of colors we might expect"""

    why: str
    """Reasons for those colors"""

    def _apply(self, block: Block) -> None:
        """Respond with a test message."""
        pass


def main() -> None:
    # Create a temporary in-memory database for the demo
    conn = connect_db(":memory:")

    # Create the LLM service with FooBar tool
    llm_service = OpenAILLMService(
        model=OpenAIModel.GPT_5,
        service_tier=OpenAIServiceTier.DEFAULT,
        tools=(FooBar,),
        system_message="You are a helpful assistant that can use tools.",
        compression_threshold=320_000,
        verbosity=OpenAIVerbosity.MEDIUM,
        reasoning_effort=OpenAIReasoningEffort.MEDIUM,
        timeout_minutes=60,
    )

    with conn.transaction_cursor() as cursor:
        # Create a conversation
        conversation = Conversation.create(cursor)

        # Add a user message
        conversation.add_user_text(
            "What color is the ocean? Please call the FooBar tool."
        )

        # Submit to LLM
        logger.info("Submitting demo conversation to LLM")
        response_id = llm_service.prompt(
            previously=None,
            new_messages=conversation.blocks,
        )

    print(f"Response ID: {response_id}")
    print("---")

    # Poll for results
    logger.info(f"Polling for response {response_id}")
    while (result := llm_service.try_fetch(response_id)) is None:
        print("Response not ready yet. Sleeping...")
        time.sleep(5)
    logger.info("Response received")

    # Process the response
    print(f"Text responses: {result.texts}")
    print(f"Tool calls: {len(result.tools)}")
    for tool in result.tools:
        print(f"  - {type(tool).__name__}: {tool}")

    # Print usage information
    print("\nToken usage:")
    print(f"  Input: {result.input_tokens}")
    print(f"  Output: {result.output_tokens}")
    print(f"  Total: {result.input_tokens + result.output_tokens}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
