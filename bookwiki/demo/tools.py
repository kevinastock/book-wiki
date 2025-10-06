"""Demo script showing tool schemas as sent to OpenAI."""

import json
from typing import Type

from bookwiki.impls.openai import GenerateJsonSchemaNoTitles
from bookwiki.tools import get_all_tools
from bookwiki.tools.base import ToolModel


def print_tool_schema(tool_class: Type[ToolModel]) -> None:
    """Pretty print the schema for a single tool as sent to OpenAI."""

    # Create the tool schema in OpenAI format
    tool_schema = {
        "type": "function",
        "name": tool_class.__name__,
        "description": tool_class.get_tool_description(),
        "parameters": tool_class.model_json_schema(
            schema_generator=GenerateJsonSchemaNoTitles
        ),
        "strict": True,
    }

    # Pretty print with indentation
    print(f"\n{'=' * 60}")
    print(f"Tool: {tool_class.__name__}")
    print(f"{'=' * 60}")
    print(json.dumps(tool_schema, indent=2))


def main() -> None:
    """Demo all tool schemas as they would be sent to OpenAI."""

    print("OpenAI Tool Schemas")
    print("=" * 70)
    print("\nThis demo shows the exact JSON schemas sent to OpenAI for each tool.")
    print("These schemas define how the LLM can call tools with proper validation.")

    # Get all registered tools
    tools = get_all_tools()

    print(f"\nFound {len(tools)} registered tools:")
    for tool in tools:
        print(f"  - {tool.__name__}")

    # Print schema for each tool
    for tool_class in tools:
        print_tool_schema(tool_class)

    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")
    print(f"Total tools: {len(tools)}")

    # Calculate total schema size
    total_size = 0
    for tool_class in tools:
        schema = {
            "type": "function",
            "name": tool_class.__name__,
            "description": tool_class.get_tool_description(),
            "parameters": tool_class.model_json_schema(
                schema_generator=GenerateJsonSchemaNoTitles
            ),
            "strict": True,
        }
        schema_str = json.dumps(schema)
        total_size += len(schema_str)

    print(f"Total schema size: {total_size:,} bytes")
    print(f"Average per tool: {total_size // len(tools):,} bytes")

    # Show the complete tools array as it would be sent to OpenAI
    print(f"\n{'=' * 60}")
    print("Complete tools array for OpenAI API:")
    print(f"{'=' * 60}")

    all_tools_schema = []
    for tool_class in tools:
        tool_schema = {
            "type": "function",
            "name": tool_class.__name__,
            "description": tool_class.get_tool_description(),
            "parameters": tool_class.model_json_schema(
                schema_generator=GenerateJsonSchemaNoTitles
            ),
            "strict": True,
        }
        all_tools_schema.append(tool_schema)

    print(json.dumps(all_tools_schema, indent=2))


if __name__ == "__main__":
    main()
