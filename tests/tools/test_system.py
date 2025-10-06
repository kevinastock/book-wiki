"""Tests for bookwiki system tools."""

from string import Template

import pytest

from bookwiki.db import SafeConnection
from bookwiki.models import Block, Conversation, Prompt
from bookwiki.tools.system import RequestExpertFeedback, SpawnAgent


def test_request_expert_feedback_apply(temp_db: SafeConnection) -> None:
    """Test RequestExpertFeedback tool application."""
    with temp_db.transaction_cursor() as cursor:
        conversation = Conversation.create(cursor)

        block = Block.create_tool_use(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            name="RequestExpertFeedback",
            use_id="feedback_1",
            params='{"request": "How should I handle this edge case?"}',
        )

        tool = RequestExpertFeedback(
            tool_id="feedback_1",
            tool_name="RequestExpertFeedback",
            request="How should I handle this edge case?",
        )

        # Apply the tool (it's a no-op)
        tool.apply(block)

        # Since it's a no-op, block should not have a response set
        updated_block = Block.get_by_id(cursor, block.id)
        assert updated_block is not None
        assert updated_block.tool_response is None


def test_spawn_agent_success(temp_db: SafeConnection) -> None:
    """Test successfully spawning an agent with valid prompt."""
    with temp_db.transaction_cursor() as cursor:
        conversation = Conversation.create(cursor)

        # Create a prompt template first
        prompt_block = Block.create_text(
            cursor,
            conversation.id,
            conversation.current_generation,
            "assistant",
            "Creating prompt",
        )
        Prompt.create(
            cursor=cursor,
            key="analyzer_agent",
            create_block_id=prompt_block.id,
            summary="Analyzer agent prompt",
            template=Template(
                "Analyze $target in chapter $chapter_num with focus on $aspect"
            ),
        )

        # Now spawn an agent using the prompt
        spawn_block = Block.create_tool_use(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            name="SpawnAgent",
            use_id="spawn_1",
            params=(
                '{"prompt_key": "analyzer_agent", '
                '"template_names": ["target", "chapter_num", "aspect"], '
                '"template_values": ["Rand", "5", "leadership"]}'
            ),
        )

        tool = SpawnAgent(
            tool_id="spawn_1",
            tool_name="SpawnAgent",
            prompt_key="analyzer_agent",
            template_names=["target", "chapter_num", "aspect"],
            template_values=["Rand", "5", "leadership"],
        )

        tool.apply(spawn_block)

        # Check no error occurred
        updated_spawn_block = Block.get_by_id(cursor, spawn_block.id)
        assert updated_spawn_block is not None
        assert updated_spawn_block.tool_response is None  # No response set
        assert updated_spawn_block.errored is False

        # Verify a new conversation was created with parent
        child_conversation = spawn_block.spawned_conversation
        assert child_conversation is not None

        # Verify the new conversation has the substituted text
        child_conversation = spawn_block.spawned_conversation
        assert child_conversation is not None
        user_blocks = [b for b in child_conversation.blocks if b.text_role == "user"]
        assert len(user_blocks) == 1
        expected_text = "Analyze Rand in chapter 5 with focus on leadership"
        assert user_blocks[0].text_body == expected_text


def test_spawn_agent_nonexistent_prompt(temp_db: SafeConnection) -> None:
    """Test spawning an agent with a nonexistent prompt key."""
    with temp_db.transaction_cursor() as cursor:
        conversation = Conversation.create(cursor)

        block = Block.create_tool_use(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            name="SpawnAgent",
            use_id="spawn_missing",
            params=(
                '{"prompt_key": "nonexistent_prompt", '
                '"template_names": ["var"], '
                '"template_values": ["value"]}'
            ),
        )

        tool = SpawnAgent(
            tool_id="spawn_missing",
            tool_name="SpawnAgent",
            prompt_key="nonexistent_prompt",
            template_names=["var"],
            template_values=["value"],
        )

        tool.apply(block)

        # Check error response
        updated_block = Block.get_by_id(cursor, block.id)
        assert updated_block is not None
        assert updated_block.tool_response == "No prompt with that key exists"
        assert updated_block.errored is True

        # Verify no new conversation was created
        child_conversation = block.spawned_conversation
        assert child_conversation is None


def test_spawn_agent_wrong_template_vars(temp_db: SafeConnection) -> None:
    """Test spawning an agent with incorrect template variables."""
    with temp_db.transaction_cursor() as cursor:
        conversation = Conversation.create(cursor)

        # Create a prompt template
        prompt_block = Block.create_text(
            cursor,
            conversation.id,
            conversation.current_generation,
            "assistant",
            "Creating prompt",
        )
        Prompt.create(
            cursor=cursor,
            key="strict_prompt",
            create_block_id=prompt_block.id,
            summary="Prompt with specific variables",
            template=Template("Process $item_a and $item_b with $method"),
        )

        # Try to spawn with wrong variables
        spawn_block = Block.create_tool_use(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            name="SpawnAgent",
            use_id="spawn_wrong_vars",
            params=(
                '{"prompt_key": "strict_prompt", '
                '"template_names": ["item_a", "wrong_var"], '
                '"template_values": ["value1", "value2"]}'
            ),
        )

        tool = SpawnAgent(
            tool_id="spawn_wrong_vars",
            tool_name="SpawnAgent",
            prompt_key="strict_prompt",
            template_names=["item_a", "wrong_var"],
            template_values=[
                "value1",
                "value2",
            ],  # Missing item_b and method, has extra wrong_var
        )

        tool.apply(spawn_block)

        # Check error response
        updated_spawn_block = Block.get_by_id(cursor, spawn_block.id)
        assert updated_spawn_block is not None
        assert updated_spawn_block.tool_response is not None
        # The error message should indicate the mismatch
        assert "Failed to substitute vars" in updated_spawn_block.tool_response
        assert "Expected keys:" in updated_spawn_block.tool_response
        assert "Actual keys:" in updated_spawn_block.tool_response
        assert updated_spawn_block.errored is True


def test_prompt_without_template_vars_succeeds(temp_db: SafeConnection) -> None:
    """Test that creating a prompt with no template variables succeeds."""
    with temp_db.transaction_cursor() as cursor:
        conversation = Conversation.create(cursor)

        # Create a prompt template with no variables
        prompt_block = Block.create_text(
            cursor,
            conversation.id,
            conversation.current_generation,
            "assistant",
            "Creating prompt",
        )

        # Creating a prompt without template variables should succeed
        prompt = Prompt.create(
            cursor=cursor,
            key="static_prompt",
            create_block_id=prompt_block.id,
            summary="Static prompt without variables",
            template=Template("This is a static prompt with no variables"),
        )

        # Verify the prompt was created successfully
        assert prompt.key == "static_prompt"
        assert prompt.summary == "Static prompt without variables"
        assert prompt.template.template == "This is a static prompt with no variables"
        assert len(prompt.template.get_identifiers()) == 0


def test_prompt_with_empty_key_raises_error(temp_db: SafeConnection) -> None:
    """Test that creating a prompt with an empty key raises an error."""
    with temp_db.transaction_cursor() as cursor:
        conversation = Conversation.create(cursor)

        # Create a prompt template with variables
        prompt_block = Block.create_text(
            cursor,
            conversation.id,
            conversation.current_generation,
            "assistant",
            "Creating prompt",
        )

        # Creating a prompt with empty key should raise ValueError
        with pytest.raises(ValueError, match="Prompt key cannot be empty"):
            Prompt.create(
                cursor=cursor,
                key="",
                create_block_id=prompt_block.id,
                summary="Prompt with variables",
                template=Template("Hello $name, you are $age years old"),
            )


def test_spawn_agent_creates_parent_child_relationship(temp_db: SafeConnection) -> None:
    """Test that SpawnAgent properly creates parent-child conversation relationship."""
    with temp_db.transaction_cursor() as cursor:
        parent_conversation = Conversation.create(cursor)

        # Create a prompt
        prompt_block = Block.create_text(
            cursor,
            parent_conversation.id,
            parent_conversation.current_generation,
            "assistant",
            "Prompt creation",
        )
        Prompt.create(
            cursor=cursor,
            key="child_prompt",
            create_block_id=prompt_block.id,
            summary="Child agent prompt",
            template=Template("Child agent task: $task"),
        )

        # Spawn child agent
        spawn_block = Block.create_tool_use(
            cursor=cursor,
            conversation_id=parent_conversation.id,
            generation=parent_conversation.current_generation,
            name="SpawnAgent",
            use_id="spawn_child",
            params=(
                '{"prompt_key": "child_prompt", '
                '"template_names": ["task"], '
                '"template_values": ["analyze data"]}'
            ),
        )

        tool = SpawnAgent(
            tool_id="spawn_child",
            tool_name="SpawnAgent",
            prompt_key="child_prompt",
            template_names=["task"],
            template_values=["analyze data"],
        )

        tool.apply(spawn_block)

        # Verify parent-child relationship
        child_conversation = spawn_block.spawned_conversation
        assert child_conversation is not None
        child_conversation_id = child_conversation.id
        assert child_conversation.parent_block_id == spawn_block.id

        # Verify the child conversation is different from parent
        assert child_conversation_id != parent_conversation.id

        # Verify the spawn block belongs to parent conversation
        assert spawn_block.conversation_id == parent_conversation.id


def test_spawn_agent_mismatched_lists(temp_db: SafeConnection) -> None:
    """Test that mismatched template_names and template_values lists raise an error."""
    with temp_db.transaction_cursor() as cursor:
        conversation = Conversation.create(cursor)

        # Create a prompt template
        prompt_block = Block.create_text(
            cursor,
            conversation.id,
            conversation.current_generation,
            "assistant",
            "Creating prompt",
        )
        Prompt.create(
            cursor=cursor,
            key="test_prompt",
            create_block_id=prompt_block.id,
            summary="Test prompt",
            template=Template("Test $var1 and $var2"),
        )

        # Create spawn block with mismatched lists
        spawn_block = Block.create_tool_use(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            name="SpawnAgent",
            use_id="spawn_mismatch",
            params=(
                '{"prompt_key": "test_prompt", '
                '"template_names": ["var1", "var2"], '
                '"template_values": ["value1"]}'  # Missing value for var2
            ),
        )

        tool = SpawnAgent(
            tool_id="spawn_mismatch",
            tool_name="SpawnAgent",
            prompt_key="test_prompt",
            template_names=["var1", "var2"],
            template_values=["value1"],  # Mismatched length
        )

        tool.apply(spawn_block)

        # Check error response
        updated_spawn_block = Block.get_by_id(cursor, spawn_block.id)
        assert updated_spawn_block is not None
        assert updated_spawn_block.tool_response is not None
        assert (
            "Variable names and values lists must have the same length"
            in updated_spawn_block.tool_response
        )
        assert updated_spawn_block.errored is True


def test_spawn_agent_dollar_sign_in_template_names(temp_db: SafeConnection) -> None:
    """Test that variable names containing $ are rejected."""
    with temp_db.transaction_cursor() as cursor:
        conversation = Conversation.create(cursor)

        # Create a prompt template
        prompt_block = Block.create_text(
            cursor,
            conversation.id,
            conversation.current_generation,
            "assistant",
            "Creating prompt",
        )
        Prompt.create(
            cursor=cursor,
            key="test_prompt",
            create_block_id=prompt_block.id,
            summary="Test prompt",
            template=Template("Test $var1"),
        )

        # Create spawn block with $ in variable name
        spawn_block = Block.create_tool_use(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            name="SpawnAgent",
            use_id="spawn_dollar",
            params=(
                '{"prompt_key": "test_prompt", '
                '"template_names": ["$var1"], '
                '"template_values": ["value1"]}'
            ),
        )

        tool = SpawnAgent(
            tool_id="spawn_dollar",
            tool_name="SpawnAgent",
            prompt_key="test_prompt",
            template_names=["$var1"],  # Should not contain $
            template_values=["value1"],
        )

        tool.apply(spawn_block)

        # Check error response
        updated_spawn_block = Block.get_by_id(cursor, spawn_block.id)
        assert updated_spawn_block is not None
        assert updated_spawn_block.tool_response is not None
        response = updated_spawn_block.tool_response
        assert "Variable name '$var1' contains '$' character" in response
        assert "Variable names should not include the '$' prefix" in response
        assert updated_spawn_block.errored is True
