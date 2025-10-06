"""Tests for bookwiki base tool classes."""

from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from bookwiki.db import SafeConnection
from bookwiki.models import Block, Conversation
from bookwiki.tools.base import LLMSolvableError, ToolModel


class ExampleTool(ToolModel):
    """Test tool for unit testing"""

    test_param: str
    """A test parameter"""

    def _apply(self, block: Block) -> None:
        """Apply the test tool."""
        if self.test_param == "error":
            raise LLMSolvableError("Test error message")
        elif self.test_param == "exception":
            raise ValueError("Unexpected error")
        else:
            block.respond(f"Test response: {self.test_param}")


class AnotherExampleTool(ToolModel):
    """Another test tool"""

    value: int

    def _apply(self, block: Block) -> None:
        block.respond(f"Value: {self.value}")


def test_tool_model_validation() -> None:
    """Test ToolModel validation and initialization."""
    # Valid initialization
    tool = ExampleTool(tool_id="test_123", tool_name="ExampleTool", test_param="hello")
    assert tool.tool_id == "test_123"
    assert tool.tool_name == "ExampleTool"
    assert tool.test_param == "hello"

    # Test tool_name validation
    with pytest.raises(ValidationError) as exc_info:
        ExampleTool(tool_id="test_456", tool_name="WrongName", test_param="test")
    assert "modelname must be" in str(exc_info.value)


def test_tool_model_context_injection() -> None:
    """Test that context values are injected when missing."""
    # Create tool with context
    context = {"tool_id": "context_id", "tool_name": "ExampleTool"}

    # This would normally be done by Pydantic during validation
    # We'll test the validator directly
    data = {"test_param": "value"}

    # Create tool using Pydantic's model_validate with context
    tool = ExampleTool.model_validate(data, context=context)

    assert tool.tool_id == "context_id"
    assert tool.tool_name == "ExampleTool"
    assert tool.test_param == "value"


def test_tool_get_name() -> None:
    """Test get_name method returns class name."""
    tool = ExampleTool(tool_id="test_id", tool_name="ExampleTool", test_param="test")
    assert tool.tool_name == "ExampleTool"

    another_tool = AnotherExampleTool(
        tool_id="another_id", tool_name="AnotherExampleTool", value=42
    )
    assert another_tool.tool_name == "AnotherExampleTool"


def test_tool_apply_success(temp_db: SafeConnection) -> None:
    """Test successful tool application."""
    with temp_db.transaction_cursor() as cursor:
        conversation = Conversation.create(cursor)
        block = Block.create_tool_use(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            name="ExampleTool",
            use_id="tool_success",
            params='{"test_param": "success"}',
        )

        tool = ExampleTool(
            tool_id="test_success", tool_name="ExampleTool", test_param="success"
        )

        # Apply the tool
        tool.apply(block)

        # Check that response was set
        updated_block = Block.get_by_id(cursor, block.id)
        assert updated_block is not None
        assert updated_block.tool_response == "Test response: success"
        assert updated_block.errored is False


def test_tool_apply_llm_solvable_error(temp_db: SafeConnection) -> None:
    """Test tool application with LLMSolvableError."""
    with temp_db.transaction_cursor() as cursor:
        conversation = Conversation.create(cursor)
        block = Block.create_tool_use(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            name="ExampleTool",
            use_id="tool_error",
            params='{"test_param": "error"}',
        )

        tool = ExampleTool(
            tool_id="test_error", tool_name="ExampleTool", test_param="error"
        )

        # Apply the tool - should catch LLMSolvableError
        tool.apply(block)

        # Check that error response was set
        updated_block = Block.get_by_id(cursor, block.id)
        assert updated_block is not None
        assert updated_block.tool_response == "Test error message"
        assert updated_block.errored is True


def test_tool_apply_unexpected_exception(temp_db: SafeConnection) -> None:
    """Test that unexpected exceptions are not caught."""
    with temp_db.transaction_cursor() as cursor:
        conversation = Conversation.create(cursor)
        block = Block.create_tool_use(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            name="ExampleTool",
            use_id="tool_exception",
            params='{"test_param": "exception"}',
        )

        tool = ExampleTool(
            tool_id="test_exception", tool_name="ExampleTool", test_param="exception"
        )

        # Apply the tool - should raise ValueError
        with pytest.raises(ValueError) as exc_info:
            tool.apply(block)
        assert str(exc_info.value) == "Unexpected error"


def test_add_to_conversation_basic(temp_db: SafeConnection) -> None:
    """Test adding a tool to a conversation creates the correct block."""
    with temp_db.transaction_cursor() as cursor:
        conv = Conversation.create(cursor)

        # Create a tool instance
        tool = ExampleTool(
            tool_id="test-tool-123",
            tool_name="ExampleTool",
            test_param="param",
        )

        # Add to conversation
        block = tool.add_to_conversation(conv)

        # Verify block was created
        assert isinstance(block, Block)
        assert block.tool_name == "ExampleTool"
        assert block.tool_use_id == "test-tool-123"

        # Verify params were serialized correctly
        assert block.tool_params is not None
        params = block.tool_params_json
        assert params["test_param"] == "param"
        # tool_id and tool_name should be excluded
        assert "tool_id" not in params
        assert "tool_name" not in params


def test_add_to_conversation_different_tools(temp_db: SafeConnection) -> None:
    """Test adding different tool types to a conversation."""
    with temp_db.transaction_cursor() as cursor:
        conv = Conversation.create(cursor)

        first_tool = ExampleTool(
            tool_id="first-123",
            tool_name="ExampleTool",
            test_param="param",
        )
        first_block = first_tool.add_to_conversation(conv)

        second_tool = AnotherExampleTool(
            tool_id="second-456",
            tool_name="AnotherExampleTool",
            value=5,
        )
        second_block = second_tool.add_to_conversation(conv)

        # Verify both blocks were created correctly
        assert first_block.tool_name == "ExampleTool"
        assert first_block.tool_params is not None

        assert second_block.tool_name == "AnotherExampleTool"
        assert second_block.tool_params is not None

        # Verify params for first tool
        first_params = first_block.tool_params_json
        assert first_params["test_param"] == "param"

        # Verify params for second tool
        second_params = second_block.tool_params_json
        assert second_params["value"] == 5


def test_add_to_conversation_preserves_order(temp_db: SafeConnection) -> None:
    """Test that multiple tools added to a conversation maintain order."""
    with temp_db.transaction_cursor() as cursor:
        conv = Conversation.create(cursor)

        # Add multiple tools
        tools = []
        blocks = []
        for i in range(3):
            tool = ExampleTool(
                tool_id=f"tool-{i}",
                tool_name="ExampleTool",
                test_param=f"param-{i}",
            )
            tools.append(tool)
            blocks.append(tool.add_to_conversation(conv))

        # Get all blocks for the conversation
        all_blocks = conv.unsent_blocks

        # Verify order is preserved
        assert len(all_blocks) == 3
        for i, block in enumerate(all_blocks):
            assert block.tool_use_id == f"tool-{i}"
            assert block.tool_params is not None
            params = block.tool_params_json
            assert params["test_param"] == f"param-{i}"
            assert len(params) == 1


def test_tool_model_not_implemented() -> None:
    """Test that base ToolModel raises NotImplementedError."""

    class IncompleteToolModel(ToolModel):
        """Incomplete tool"""

        # No _apply method implemented

    tool = IncompleteToolModel(tool_id="incomplete", tool_name="IncompleteToolModel")

    # Create a mock block
    mock_block = MagicMock(spec=Block)

    # Should raise NotImplementedError
    with pytest.raises(NotImplementedError) as exc_info:
        tool._apply(mock_block)
    assert "apply() not implemented for IncompleteToolModel" in str(exc_info.value)


def test_tool_model_config() -> None:
    """Test that ToolModel uses attribute docstrings."""
    # The model_config should enable use_attribute_docstrings
    assert ExampleTool.model_config.get("use_attribute_docstrings") is True

    # Check that the tool has a description
    assert ExampleTool.get_tool_description() == "Test tool for unit testing"
    assert AnotherExampleTool.get_tool_description() == "Another test tool"
