"""Demo program showing weather tool usage with OpenAI LLM service."""

import argparse
import logging
import time
from typing import Any, Dict

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


class GetWeather(ToolModel):
    """Get current weather information for a specified location."""

    location: str
    """The location to get weather for (city, state/country)"""

    units: str = "celsius"
    """Temperature units: celsius or fahrenheit"""

    def _apply(self, block: Block) -> None:
        """Respond with mock weather data."""
        # Mock weather data based on location
        weather_data: Dict[str, Dict[str, Any]] = {
            "new york": {"temp": 22, "condition": "Sunny", "humidity": 65},
            "london": {"temp": 15, "condition": "Cloudy", "humidity": 78},
            "tokyo": {"temp": 28, "condition": "Partly cloudy", "humidity": 71},
            "sydney": {"temp": 18, "condition": "Rainy", "humidity": 85},
            "paris": {"temp": 20, "condition": "Overcast", "humidity": 73},
        }

        location_key = self.location.lower()

        # Check if we have data for this location
        if location_key in weather_data:
            data = weather_data[location_key]
            temp_celsius: int = data["temp"]  # type: ignore

            # Convert temperature if requested
            temp: int
            if self.units.lower() == "fahrenheit":
                temp = round((temp_celsius * 9 / 5) + 32)
                unit_symbol = "°F"
            else:
                temp = temp_celsius
                unit_symbol = "°C"

            response = (
                f"Current weather in {self.location.title()}:\n"
                f"Temperature: {temp}{unit_symbol}\n"
                f"Condition: {data['condition']}\n"
                f"Humidity: {data['humidity']}%"
            )
        else:
            response = (
                f"Weather data not available for {self.location}. "
                f"Try: New York, London, Tokyo, Sydney, or Paris."
            )

        block.respond(response)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Weather Demo - Testing LLM with GetWeather tool"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose debug logging"
    )
    args = parser.parse_args()

    # Set up logging based on verbose flag
    if args.verbose:
        # Enable debug logging for detailed output
        logging.basicConfig(
            level=logging.DEBUG, format="%(name)s - %(levelname)s - %(message)s"
        )
        logger.info("Verbose mode enabled - showing debug logging")
    else:
        logging.basicConfig(level=logging.INFO)

    print("Weather Demo - Testing LLM with GetWeather tool")
    print("=" * 50)
    if args.verbose:
        print("Running in verbose mode with debug logging")
        print("=" * 50)

    # Create a temporary in-memory database for the demo
    conn = connect_db(":memory:")

    # Create the LLM service with weather tool
    llm_service = OpenAILLMService(
        model=OpenAIModel.GPT_5,
        service_tier=OpenAIServiceTier.DEFAULT,
        tools=(GetWeather,),
        system_message=(
            "You are a helpful weather assistant. "
            "When asked about weather, use the GetWeather tool."
        ),
        compression_threshold=320_000,
        verbosity=OpenAIVerbosity.MEDIUM,
        reasoning_effort=OpenAIReasoningEffort.MEDIUM,
        timeout_minutes=60,
    )

    with conn.transaction_cursor() as cursor:
        # Create a conversation
        conversation = Conversation.create(cursor)

        # Add a user message asking for weather
        conversation.add_user_text(
            "What's the weather like in Tokyo? Please show it in Fahrenheit."
        )

        # Submit to LLM
        logger.info("Submitting weather query to LLM")
        response_id = llm_service.prompt(
            previously=None,
            new_messages=conversation.unsent_blocks,
        )
        conversation.mark_all_blocks_as_sent()

    print(f"Response ID: {response_id}")
    print("Polling for response...")

    # Poll for results
    start_time = time.time()
    while (result := llm_service.try_fetch(response_id)) is None:
        elapsed = time.time() - start_time
        print(f"Response not ready yet. Waiting... ({elapsed:.1f}s)")
        time.sleep(2)

        # Timeout after 60 seconds
        if elapsed > 60:
            print("Timeout waiting for response")
            return

    logger.info("Response received")

    # Process and display the response
    print("\n" + "=" * 50)
    print("LLM RESPONSE:")
    print("=" * 50)

    if result.texts:
        for text in result.texts:
            print(f"Assistant: {text}")

    if result.tools:
        print(f"\nTool calls made: {len(result.tools)}")
        for i, tool in enumerate(result.tools, 1):
            print(f"\n{i}. {type(tool).__name__}:")
            if isinstance(tool, GetWeather):
                print(f"   Location: {tool.location}")
                print(f"   Units: {tool.units}")

    # Execute the tools and add their responses to the conversation
    print("\n" + "=" * 50)
    print("EXECUTING TOOLS:")
    print("=" * 50)

    with conn.transaction_cursor() as cursor:
        # Get the conversation from the database with the new cursor
        row = cursor.execute(
            "SELECT * FROM conversation WHERE id = ?", (conversation.id,)
        ).fetchone()

        if row:
            conversation_with_cursor = Conversation(
                _cursor=cursor,
                id=row["id"],
                previously=row["previously"],
                parent_block_id=row["parent_block_id"],
                total_input_tokens=row["total_input_tokens"],
                total_output_tokens=row["total_output_tokens"],
                current_tokens=row["current_tokens"],
                current_generation=row["current_generation"],
                waiting_on_id=row["waiting_on_id"],
            )

            # Add assistant message with tool calls first
            conversation_with_cursor.add_assistant_text(
                "\n".join(result.texts) if result.texts else ""
            )

            for tool in result.tools:
                # Create a tool use block and apply the tool
                block = tool.add_to_conversation(conversation_with_cursor)
                tool.apply(block)

                # Get and display the tool response
                cursor.execute(
                    "SELECT tool_response FROM block WHERE id = ?", (block.id,)
                )
                response_row = cursor.fetchone()
                if response_row and response_row["tool_response"]:
                    print(f"{type(tool).__name__} executed successfully")

        # Update conversation state with token usage
        conversation_with_cursor.update_tokens(
            result.input_tokens, result.output_tokens
        )

    # Send the tool results back to the LLM for a final response
    print("\n" + "=" * 50)
    print("SENDING TOOL RESULTS BACK TO LLM:")
    print("=" * 50)

    with conn.transaction_cursor() as cursor:
        # Get updated conversation
        row = cursor.execute(
            "SELECT * FROM conversation WHERE id = ?", (conversation.id,)
        ).fetchone()

        if row:
            updated_conversation = Conversation(
                _cursor=cursor,
                id=row["id"],
                previously=row["previously"],
                parent_block_id=row["parent_block_id"],
                total_input_tokens=row["total_input_tokens"],
                total_output_tokens=row["total_output_tokens"],
                current_tokens=row["current_tokens"],
                current_generation=row["current_generation"],
                waiting_on_id=row["waiting_on_id"],
            )

            # Submit the updated conversation (including tool results) back to LLM
            response_id_2 = llm_service.prompt(
                previously=result.updated_prev,
                new_messages=updated_conversation.unsent_blocks,
            )

    print(f"Second response ID: {response_id_2}")
    print("Polling for final response...")

    # Poll for the final response
    start_time = time.time()
    while (final_result := llm_service.try_fetch(response_id_2)) is None:
        elapsed = time.time() - start_time
        print(f"Final response not ready yet. Waiting... ({elapsed:.1f}s)")
        time.sleep(2)

        if elapsed > 60:
            print("Timeout waiting for final response")
            return

    print("\n" + "=" * 50)
    print("FINAL LLM RESPONSE:")
    print("=" * 50)

    if final_result.texts:
        for text in final_result.texts:
            print(f"Assistant: {text}")

    # Print combined usage information
    print("\n" + "=" * 50)
    print("TOKEN USAGE:")
    print("=" * 50)
    print(f"1st: I={result.input_tokens}, O={result.output_tokens}")
    print(f"2nd: I={final_result.input_tokens}, O={final_result.output_tokens}")


if __name__ == "__main__":
    main()
