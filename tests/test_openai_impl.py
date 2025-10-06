"""Tests for OpenAI LLM service implementation."""

from typing import Any, Dict, List, cast

from openai._types import NOT_GIVEN

from bookwiki.config_enums import (
    OpenAIModel,
    OpenAIReasoningEffort,
    OpenAIServiceTier,
    OpenAIVerbosity,
)
from bookwiki.impls.openai import OpenAILLMService
from bookwiki.models import Block
from bookwiki.tools.base import ToolModel


class MockTool(ToolModel):
    """A mock tool for testing schema generation"""

    test_string: str
    """A test string parameter"""

    test_int: int
    """A test integer parameter"""

    test_bool: bool = False
    """A test boolean parameter with default"""

    def _apply(self, block: Block) -> None:
        pass


class AnotherMockTool(ToolModel):
    """Another mock tool for testing"""

    required_param: str
    """A required parameter"""

    optional_param: int = 42
    """An optional parameter with default value"""

    def _apply(self, block: Block) -> None:
        pass


def test_create_tools_from_models_single_tool() -> None:
    """Test creating OpenAI tool parameters from a single ToolModel."""
    tool_models = (MockTool,)

    result = OpenAILLMService._create_tools_from_models(tool_models)

    assert len(result) == 1
    tool_param = result[0]

    # Check basic structure
    assert tool_param["type"] == "function"
    assert tool_param["name"] == "MockTool"
    assert tool_param["description"] == "A mock tool for testing schema generation"
    assert tool_param["strict"] is True

    # Check parameters schema structure
    schema = tool_param["parameters"]
    assert isinstance(schema, dict)
    assert "type" in schema
    assert schema["type"] == "object"
    assert "properties" in schema
    assert "required" in schema

    # Check that required fields include all tool-specific fields for OpenAI strict mode
    # In strict mode, ALL fields must be in required array, even those with defaults
    required_fields = cast(List[str], schema["required"])
    # tool_id and tool_name should NOT appear here (excluded from schema)
    assert "tool_id" not in required_fields
    assert "tool_name" not in required_fields
    assert "test_string" in required_fields
    assert "test_int" in required_fields
    # In OpenAI strict mode, even fields with defaults must be required
    assert "test_bool" in required_fields

    # Check property definitions - only tool-specific fields should be present
    properties = cast(Dict[str, Any], schema["properties"])
    assert "tool_id" not in properties
    assert "tool_name" not in properties
    assert "test_string" in properties
    assert "test_int" in properties
    assert "test_bool" in properties

    # Check that string properties have correct type
    assert cast(Dict[str, Any], properties["test_string"])["type"] == "string"
    assert cast(Dict[str, Any], properties["test_int"])["type"] == "integer"
    assert cast(Dict[str, Any], properties["test_bool"])["type"] == "boolean"

    # Check that docstrings appear as descriptions for properties
    assert (
        cast(Dict[str, Any], properties["test_string"])["description"]
        == "A test string parameter"
    )
    assert (
        cast(Dict[str, Any], properties["test_int"])["description"]
        == "A test integer parameter"
    )
    assert (
        cast(Dict[str, Any], properties["test_bool"])["description"]
        == "A test boolean parameter with default"
    )


def test_create_tools_from_models_multiple_tools() -> None:
    """Test creating OpenAI tool parameters from multiple ToolModels."""
    tool_models = (MockTool, AnotherMockTool)

    result = OpenAILLMService._create_tools_from_models(tool_models)

    assert len(result) == 2

    # Find tools by name
    mock_tool = next(t for t in result if t["name"] == "MockTool")
    another_tool = next(t for t in result if t["name"] == "AnotherMockTool")

    # Check MockTool
    assert mock_tool["description"] == "A mock tool for testing schema generation"

    # Check AnotherMockTool
    assert another_tool["description"] == "Another mock tool for testing"

    # Verify required fields for AnotherMockTool
    # In OpenAI strict mode, ALL fields must be in required array
    another_schema = cast(Dict[str, Any], another_tool["parameters"])
    required_fields = cast(List[str], another_schema["required"])
    assert "required_param" in required_fields
    assert (
        "optional_param" in required_fields
    )  # In strict mode, even fields with defaults are required


def test_create_tools_from_models_empty() -> None:
    """Test creating tools from empty tuple."""
    result = OpenAILLMService._create_tools_from_models(())
    assert result == []


def test_create_tools_schema_excludes_base_fields_from_extra() -> None:
    """Test that model_json_schema() doesn't include extra base ToolModel fields."""
    tool_models = (MockTool,)

    result = OpenAILLMService._create_tools_from_models(tool_models)
    schema = result[0]["parameters"]

    # The schema should include the base fields (tool_id, tool_name)
    # but they shouldn't appear as "extra" properties beyond what's expected
    properties = cast(Dict[str, Any], schema)["properties"]
    properties_dict = cast(Dict[str, Any], properties)

    # Check that we have the expected fields (excluding base ToolModel fields)
    expected_fields = {"test_string", "test_int", "test_bool"}
    actual_fields = set(properties_dict.keys())

    # We should have exactly our expected fields, no more no less
    # (allowing for potential additional Pydantic metadata fields)
    for field in expected_fields:
        assert field in actual_fields, f"Expected field {field} not found in schema"

    # Ensure base fields are NOT present
    base_fields = {"tool_id", "tool_name"}
    for field in base_fields:
        assert field not in actual_fields, f"Base field {field} should not be in schema"


def test_openai_service_tool_initialization() -> None:
    """Test that OpenAILLMService correctly initializes with tools."""
    service = OpenAILLMService(
        model=OpenAIModel.GPT_5,
        service_tier=OpenAIServiceTier.DEFAULT,
        tools=(MockTool, AnotherMockTool),
        system_message="Test system message",
        compression_threshold=320_000,
        verbosity=OpenAIVerbosity.MEDIUM,
        reasoning_effort=OpenAIReasoningEffort.MEDIUM,
        timeout_minutes=60,
    )

    # Should have created tool parameters
    assert service._tool_params is not None
    tool_params = cast(List[Any], service._tool_params)
    assert len(tool_params) == 2

    # Should have set tool_choice to auto
    assert service._tool_choice == "auto"


def test_openai_service_no_tools_initialization() -> None:
    """Test that OpenAILLMService correctly initializes without tools."""
    service = OpenAILLMService(
        model=OpenAIModel.GPT_5,
        service_tier=OpenAIServiceTier.DEFAULT,
        tools=(),
        system_message="",
        compression_threshold=320_000,
        verbosity=OpenAIVerbosity.MEDIUM,
        reasoning_effort=OpenAIReasoningEffort.MEDIUM,
        timeout_minutes=60,
    )

    # Should not have tool parameters
    assert service._tool_params is NOT_GIVEN
    assert service._tool_choice is NOT_GIVEN


def test_tool_schema_format_matches_openai_docs() -> None:
    """Test that generated schema format matches OpenAI API documentation structure."""
    tool_models = (MockTool,)

    result = OpenAILLMService._create_tools_from_models(tool_models)
    tool_param = result[0]

    # Check top-level structure matches OpenAI function tool format
    expected_top_level_keys = {"type", "name", "description", "parameters", "strict"}
    assert set(tool_param.keys()) == expected_top_level_keys

    # Check parameters object structure
    params = cast(Dict[str, Any], tool_param["parameters"])
    expected_param_keys = {"type", "properties", "required"}
    assert all(key in params for key in expected_param_keys)

    # Check that each property has a type
    properties_dict = cast(Dict[str, Any], params["properties"])
    for prop_name, prop_def in properties_dict.items():
        prop_dict = cast(Dict[str, Any], prop_def)
        assert "type" in prop_dict, f"Property {prop_name} missing type"
        # Type should be a valid JSON schema type
        valid_types = {"string", "number", "integer", "boolean", "array", "object"}
        assert prop_dict["type"] in valid_types


def test_weather_tool_matches_openai_docs_example() -> None:
    """Test that a weather tool matches the exact format from OpenAI docs."""

    class GetWeatherTool(ToolModel):
        """Get current temperature for a given location."""

        location: str
        """City and country e.g. Bogotá, Colombia"""

    tool_models = (GetWeatherTool,)
    result = OpenAILLMService._create_tools_from_models(tool_models)

    assert len(result) == 1
    tool_param = result[0]

    # Create the expected structure exactly as it appears in OpenAI docs
    expected_tool = {
        "type": "function",
        "name": "GetWeatherTool",
        "description": "Get current temperature for a given location.",
        "parameters": {
            "properties": {
                "location": {
                    "description": "City and country e.g. Bogotá, Colombia",
                    "type": "string",
                }
            },
            "required": ["location"],
            "type": "object",
            "additionalProperties": False,
        },
        "strict": True,
    }

    # The generated tool should match exactly
    assert tool_param == expected_tool, (
        f"Generated tool does not match expected structure.\n"
        f"Generated: {tool_param}\nExpected: {expected_tool}"
    )
