"""Tool registry and dispatch tests for bookwiki."""

import json

import pytest
from pydantic import ValidationError

from bookwiki.db import SafeConnection
from bookwiki.models import Block, Conversation
from bookwiki.tools import (
    ToolModel,
    get_all_tools,
)
from bookwiki.tools.base import LLMSolvableError
from bookwiki.tools.chapter import ReadChapter
from bookwiki.tools.wiki import ReadWikiPage, WriteWikiPage


def test_tool_registration_and_lookup() -> None:
    """Test that all tools are properly registered and can be retrieved."""
    all_tools = get_all_tools()

    # Verify we have tools registered
    assert len(all_tools) > 0

    # Verify all tools are ToolModel subclasses
    for tool_class in all_tools:
        assert issubclass(tool_class, ToolModel)
        assert hasattr(tool_class, "get_tool_description")
        assert isinstance(tool_class.get_tool_description(), str)


def test_tool_parameter_validation_and_injection() -> None:
    """Test tool parameter validation and context injection."""
    # Test successful validation with explicit parameters
    tool = ReadChapter(tool_id="test_123", tool_name="ReadChapter", chapter_offset=-1)
    assert tool.tool_id == "test_123"
    assert tool.tool_name == "ReadChapter"
    assert tool.chapter_offset == -1
    assert tool.tool_name == "ReadChapter"

    # Test context injection
    context = {"tool_id": "context_id", "tool_name": "ReadChapter"}
    data = {"chapter_offset": -5}

    injected_tool = ReadChapter.model_validate(data, context=context)
    assert injected_tool.tool_id == "context_id"
    assert injected_tool.tool_name == "ReadChapter"
    assert injected_tool.chapter_offset == -5

    # Test tool_name validation failure
    with pytest.raises(ValidationError) as exc_info:
        ReadChapter(tool_id="test_456", tool_name="WrongToolName", chapter_offset=0)
    assert "modelname must be" in str(exc_info.value)


def test_tool_union_type_dispatch() -> None:
    """Test that the Tool union type can dispatch to correct tool classes."""
    # Test ReadChapter dispatch
    read_data = {"chapter_offset": -1}
    context = {"tool_id": "read_1", "tool_name": "ReadChapter"}

    # This should validate as a ReadChapter
    read_tool = ReadChapter.model_validate(read_data, context=context)
    assert isinstance(read_tool, ReadChapter)
    assert read_tool.chapter_offset == -1

    # Test WriteWikiPage dispatch
    write_data = {
        "slug": "test-page",
        "title": "Test Page",
        "names": ["Test"],
        "summary": "A test page",
        "body": "Test content",
        "create": True,
    }
    context = {"tool_id": "write_1", "tool_name": "WriteWikiPage"}

    write_tool = WriteWikiPage.model_validate(write_data, context=context)
    assert isinstance(write_tool, WriteWikiPage)
    assert write_tool.slug == "test-page"
    assert write_tool.create is True

    # Test ReadWikiPage dispatch
    read_wiki_data = {
        "slug": "test-page",
    }
    context = {"tool_id": "read_wiki_1", "tool_name": "ReadWikiPage"}

    read_wiki_tool = ReadWikiPage.model_validate(read_wiki_data, context=context)
    assert isinstance(read_wiki_tool, ReadWikiPage)
    assert read_wiki_tool.slug == "test-page"


def test_tool_error_propagation(temp_db: SafeConnection) -> None:
    """Test that tool errors are properly handled and propagated."""
    with temp_db.transaction_cursor() as cursor:
        conversation = Conversation.create(cursor)

        # Create a tool use block for a non-existent chapter
        block = Block.create_tool_use(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            name="ReadChapter",
            use_id="read_error",
            params='{"chapter_offset": 999}',
        )

        # Create tool and apply it
        tool = ReadChapter(
            tool_id="read_error", tool_name="ReadChapter", chapter_offset=999
        )

        # This should trigger an LLMSolvableError which gets caught
        # and converted to error response
        tool.apply(block)

        # Check that error was properly recorded
        updated_block = Block.get_by_id(cursor, block.id)
        assert updated_block is not None
        assert updated_block.tool_response is not None
        assert "Cannot read future chapters" in updated_block.tool_response
        assert updated_block.errored is True


def test_tool_dependency_injection_patterns() -> None:
    """Test patterns for tool dependency injection and parameter handling."""
    # Test tool creation from block parameters
    params_data = {"chapter_offset": -3}

    # Simulate how tools might be created from stored block parameters
    context = {"tool_id": "from_block", "tool_name": "ReadChapter"}

    tool = ReadChapter.model_validate(params_data, context=context)
    assert tool.tool_id == "from_block"
    assert tool.tool_name == "ReadChapter"
    assert tool.chapter_offset == -3

    # Test complex tool with many parameters
    wiki_params = {
        "slug": "complex-character",
        "title": "Complex Character",
        "names": ["Character", "The Complex One"],
        "summary": "A very complex character",
        "body": "This character has many facets and depths.",
        "create": False,
    }

    wiki_context = {"tool_id": "complex_write", "tool_name": "WriteWikiPage"}

    wiki_tool = WriteWikiPage.model_validate(wiki_params, context=wiki_context)
    assert wiki_tool.tool_id == "complex_write"
    assert wiki_tool.names == ["Character", "The Complex One"]
    assert wiki_tool.create is False


def test_tool_validation_edge_cases() -> None:
    """Test edge cases in tool validation."""
    # Test invalid model name (this should fail validation)
    with pytest.raises(ValidationError):
        WriteWikiPage(
            tool_id="test",
            tool_name="WrongToolName",  # Wrong model name should fail
            slug="test",
            title="Test",
            names=["Test"],
            summary="Test",
            body="Test",
            create=True,
        )

    # Test None for optional parameters
    wiki_tool = WriteWikiPage(
        tool_id="partial",
        tool_name="WriteWikiPage",
        slug="partial-page",
        title=None,  # Optional for updates
        names=None,  # Optional for updates
        summary="Updated summary",
        body=None,  # Optional for updates
        create=False,
    )
    assert wiki_tool.title is None
    assert wiki_tool.names is None
    assert wiki_tool.body is None
    assert wiki_tool.summary == "Updated summary"


def test_tool_polymorphic_behavior(temp_db: SafeConnection) -> None:
    """Test polymorphic behavior when working with tools."""
    with temp_db.transaction_cursor() as cursor:
        conversation = Conversation.create(cursor)

        # Create different types of tool blocks
        tools_and_blocks: list[tuple[ToolModel, Block]] = []

        # ReadChapter tool
        read_block = Block.create_tool_use(
            cursor,
            conversation.id,
            conversation.current_generation,
            "ReadChapter",
            "read_1",
            '{"chapter_offset": 0}',
        )
        read_tool = ReadChapter(
            tool_id="read_1", tool_name="ReadChapter", chapter_offset=0
        )
        tools_and_blocks.append((read_tool, read_block))

        # WriteWikiPage tool
        write_params = {
            "slug": "test",
            "title": "Test",
            "names": ["Test"],
            "summary": "Test",
            "body": "Test",
            "create": True,
        }
        write_block = Block.create_tool_use(
            cursor,
            conversation.id,
            conversation.current_generation,
            "WriteWikiPage",
            "write_1",
            json.dumps(write_params),
        )
        write_tool = WriteWikiPage.model_validate(
            write_params,
            context={"tool_id": "write_1", "tool_name": "WriteWikiPage"},
        )
        tools_and_blocks.append((write_tool, write_block))

        # Test polymorphic behavior
        for tool, _block in tools_and_blocks:
            assert isinstance(tool, ToolModel)
            assert tool.tool_name in ["ReadChapter", "WriteWikiPage"]
            assert tool.tool_id in ["read_1", "write_1"]
            assert hasattr(tool, "apply")

            # Each tool should have its specific attributes
            if isinstance(tool, ReadChapter):
                assert hasattr(tool, "chapter_offset")
                assert tool.chapter_offset == 0
            elif isinstance(tool, WriteWikiPage):
                assert hasattr(tool, "slug")
                assert hasattr(tool, "create")
                assert tool.slug == "test"


def test_tool_json_serialization_roundtrip() -> None:
    """Test that tools can be serialized to JSON and back."""
    # Create a complex tool
    original_tool = WriteWikiPage(
        tool_id="json_test",
        tool_name="WriteWikiPage",
        slug="json-character",
        title="JSON Character",
        names=["JSON", "The Serializable"],
        summary="A character that can be serialized",
        body="This character exists in JSON form.",
        create=True,
    )

    # Serialize to JSON
    json_data = original_tool.model_dump_json()
    parsed_data = json.loads(json_data)

    # Verify JSON structure
    assert parsed_data["slug"] == "json-character"
    assert parsed_data["names"] == ["JSON", "The Serializable"]

    # Deserialize back to tool
    context = {"tool_id": "json_test", "tool_name": "WriteWikiPage"}

    reconstructed_tool = WriteWikiPage.model_validate(parsed_data, context=context)

    # Verify reconstruction
    assert reconstructed_tool.tool_id == original_tool.tool_id
    assert reconstructed_tool.tool_name == original_tool.tool_name
    assert reconstructed_tool.slug == original_tool.slug
    assert reconstructed_tool.names == original_tool.names
    assert reconstructed_tool.summary == original_tool.summary
    assert reconstructed_tool.body == original_tool.body
    assert reconstructed_tool.create == original_tool.create


def test_custom_tool_implementation_pattern() -> None:
    """Test the pattern for implementing custom tools."""

    class TestCustomTool(ToolModel):
        """A test tool for validating custom implementations"""

        test_param: str
        """A parameter for testing"""

        numeric_param: int = 42
        """A numeric parameter with default"""

        def _apply(self, block: Block) -> None:
            if self.test_param == "error":
                raise LLMSolvableError("Custom tool error")
            elif self.test_param == "crash":
                raise RuntimeError("Unexpected error")
            else:
                block.respond(f"Custom tool executed: {self.test_param}")

    # Test normal construction
    tool = TestCustomTool(
        tool_id="custom_1",
        tool_name="TestCustomTool",
        test_param="success",
        numeric_param=100,
    )

    assert tool.tool_name == "TestCustomTool"
    assert tool.test_param == "success"
    assert tool.numeric_param == 100

    # Test default parameter
    tool_default = TestCustomTool(
        tool_id="custom_2", tool_name="TestCustomTool", test_param="default_test"
    )
    assert tool_default.numeric_param == 42

    # Test validation error for wrong model name
    with pytest.raises(ValidationError):
        TestCustomTool(tool_id="custom_3", tool_name="WrongName", test_param="fail")
