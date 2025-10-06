"""Tests for bookwiki prompt tools."""

import time
from string import Template

from bookwiki.db import SafeConnection
from bookwiki.models import Block, Conversation, Prompt
from bookwiki.tools.prompt import ListPrompts, ShowPrompt, WritePrompt


def test_list_prompts_empty(temp_db: SafeConnection) -> None:
    """Test listing prompts when none exist."""
    with temp_db.transaction_cursor() as cursor:
        conversation = Conversation.create(cursor)
        block = Block.create_tool_use(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            name="ListPrompts",
            use_id="list_empty",
            params="{}",
        )

        tool = ListPrompts(tool_id="list_empty", tool_name="ListPrompts")

        tool.apply(block)

        # Check response
        updated_block = Block.get_by_id(cursor, block.id)
        assert updated_block is not None
        assert updated_block.tool_response == "There are no stored prompts."
        assert updated_block.errored is False


def test_list_prompts_with_prompts(temp_db: SafeConnection) -> None:
    """Test listing prompts when some exist."""
    with temp_db.transaction_cursor() as cursor:
        conversation = Conversation.create(cursor)

        # Create some prompts first
        create_block1 = Block.create_text(
            cursor,
            conversation.id,
            conversation.current_generation,
            "assistant",
            "Creating prompt 1",
        )
        create_block2 = Block.create_text(
            cursor,
            conversation.id,
            conversation.current_generation,
            "assistant",
            "Creating prompt 2",
        )

        Prompt.create(
            cursor=cursor,
            key="analyzer",
            create_block_id=create_block1.id,
            summary="Analyzes text",
            template=Template("Analyze $text"),
        )

        # Small delay for different timestamp
        time.sleep(0.001)

        Prompt.create(
            cursor=cursor,
            key="summarizer",
            create_block_id=create_block2.id,
            summary="Summarizes content",
            template=Template("Summarize $content in $length words"),
        )

        # Now list them
        list_block = Block.create_tool_use(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            name="ListPrompts",
            use_id="list_full",
            params="{}",
        )

        tool = ListPrompts(tool_id="list_full", tool_name="ListPrompts")

        tool.apply(list_block)

        # Check response contains both prompts
        updated_list_block = Block.get_by_id(cursor, list_block.id)
        assert updated_list_block is not None
        assert updated_list_block.tool_response is not None
        assert "Key: analyzer" in updated_list_block.tool_response
        assert "Summary: Analyzes text" in updated_list_block.tool_response
        assert "Key: summarizer" in updated_list_block.tool_response
        assert "Summary: Summarizes content" in updated_list_block.tool_response
        assert updated_list_block.errored is False


def test_show_prompt_success(temp_db: SafeConnection) -> None:
    """Test showing an existing prompt."""
    with temp_db.transaction_cursor() as cursor:
        conversation = Conversation.create(cursor)

        # Create a prompt first
        create_block = Block.create_text(
            cursor,
            conversation.id,
            conversation.current_generation,
            "assistant",
            "Creating",
        )
        Prompt.create(
            cursor=cursor,
            key="test_prompt",
            create_block_id=create_block.id,
            summary="Test prompt for testing",
            template=Template("Test template with $variable and $another"),
        )

        # Show it
        show_block = Block.create_tool_use(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            name="ShowPrompt",
            use_id="show_test",
            params='{"key": "test_prompt"}',
        )

        tool = ShowPrompt(
            tool_id="show_test", tool_name="ShowPrompt", key="test_prompt"
        )

        tool.apply(show_block)

        # Check response
        updated_show_block = Block.get_by_id(cursor, show_block.id)
        assert updated_show_block is not None
        assert updated_show_block.tool_response is not None
        # Verify the actual template text is shown, not just the object representation
        assert "Summary: Test prompt for testing" in updated_show_block.tool_response
        assert (
            "Template: Test template with $variable and $another"
            in updated_show_block.tool_response
        )
        assert "Variables: ['another', 'variable']" in updated_show_block.tool_response
        assert updated_show_block.errored is False


def test_show_prompt_displays_template_content(temp_db: SafeConnection) -> None:
    """Test that ShowPrompt displays the actual template text, not object repr."""
    with temp_db.transaction_cursor() as cursor:
        conversation = Conversation.create(cursor)

        # Create a prompt with a complex template
        create_block = Block.create_text(
            cursor,
            conversation.id,
            conversation.current_generation,
            "assistant",
            "Creating complex prompt",
        )
        complex_template = """## Task
Analyze the character $character_name in chapter $chapter_number.

## Instructions
1. Review the character's actions
2. Note any development in $trait
3. Update the wiki page for $character_name

## Expected Output
A comprehensive analysis of $character_name with focus on $trait development."""

        Prompt.create(
            cursor=cursor,
            key="complex_analyzer",
            create_block_id=create_block.id,
            summary="Complex character analyzer",
            template=Template(complex_template),
        )

        # Show the prompt
        show_block = Block.create_tool_use(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            name="ShowPrompt",
            use_id="show_complex",
            params='{"key": "complex_analyzer"}',
        )

        tool = ShowPrompt(
            tool_id="show_complex", tool_name="ShowPrompt", key="complex_analyzer"
        )

        tool.apply(show_block)

        # Check the response contains the actual template text
        updated_show_block = Block.get_by_id(cursor, show_block.id)
        assert updated_show_block is not None
        assert updated_show_block.tool_response is not None
        response = updated_show_block.tool_response

        # Verify summary is present
        assert "Summary: Complex character analyzer" in response

        # Verify variables are listed
        assert "Variables: ['chapter_number', 'character_name', 'trait']" in response

        # Verify the actual template content is displayed
        assert "## Task" in response
        assert (
            "Analyze the character $character_name in chapter $chapter_number"
            in response
        )
        assert "## Instructions" in response
        assert "1. Review the character's actions" in response
        assert "2. Note any development in $trait" in response
        assert "3. Update the wiki page for $character_name" in response
        assert "## Expected Output" in response
        assert (
            "A comprehensive analysis of $character_name "
            "with focus on $trait development" in response
        )

        # Ensure we're NOT getting a Python object representation
        assert "<string.Template object at" not in response
        assert "Template object" not in response.lower()


def test_show_prompt_nonexistent(temp_db: SafeConnection) -> None:
    """Test showing a prompt that doesn't exist."""
    with temp_db.transaction_cursor() as cursor:
        conversation = Conversation.create(cursor)

        block = Block.create_tool_use(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            name="ShowPrompt",
            use_id="show_missing",
            params='{"key": "nonexistent_key"}',
        )

        tool = ShowPrompt(
            tool_id="show_missing", tool_name="ShowPrompt", key="nonexistent_key"
        )

        tool.apply(block)

        # Check error response
        updated_block = Block.get_by_id(cursor, block.id)
        assert updated_block is not None
        assert updated_block.tool_response == "Key nonexistent_key does not exist."
        assert updated_block.errored is True


def test_write_prompt_valid_template(temp_db: SafeConnection) -> None:
    """Test writing a prompt with a valid template."""
    with temp_db.transaction_cursor() as cursor:
        conversation = Conversation.create(cursor)

        block = Block.create_tool_use(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            name="WritePrompt",
            use_id="write_valid",
            params=(
                '{"key": "new_prompt", '
                '"summary": "A new test prompt", '
                '"template": "Analyze $character in chapter $chapter"}'
            ),
        )

        tool = WritePrompt(
            tool_id="write_valid",
            tool_name="WritePrompt",
            key="new_prompt",
            summary="A new test prompt",
            template="Analyze $character in chapter $chapter",
        )

        tool.apply(block)

        # Check success response
        updated_block = Block.get_by_id(cursor, block.id)
        assert updated_block is not None
        assert updated_block.tool_response == "Prompt stored"
        assert updated_block.errored is False

        # Verify prompt was actually stored
        stored_prompt = Prompt.get_prompt(cursor, "new_prompt")
        assert stored_prompt is not None
        assert stored_prompt.summary == "A new test prompt"
        assert (
            stored_prompt.template.template == "Analyze $character in chapter $chapter"
        )


def test_write_prompt_invalid_template(temp_db: SafeConnection) -> None:
    """Test writing a prompt with an invalid template."""
    with temp_db.transaction_cursor() as cursor:
        conversation = Conversation.create(cursor)

        block = Block.create_tool_use(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            name="WritePrompt",
            use_id="write_invalid",
            params=(
                '{"key": "bad_prompt", '
                '"summary": "Bad template test", '
                '"template": "Invalid ${ template"}'
            ),
        )

        tool = WritePrompt(
            tool_id="write_invalid",
            tool_name="WritePrompt",
            key="bad_prompt",
            summary="Bad template test",
            template="Invalid ${ template",  # Invalid template syntax
        )

        tool.apply(block)

        # Check error response
        updated_block = Block.get_by_id(cursor, block.id)
        assert updated_block is not None
        assert updated_block.tool_response == "Template is not valid, prompt rejected."
        assert updated_block.errored is True

        # Verify prompt was NOT stored
        stored_prompt = Prompt.get_prompt(cursor, "bad_prompt")
        assert stored_prompt is None


def test_write_prompt_overwrites_existing(temp_db: SafeConnection) -> None:
    """Test that writing a prompt with existing key creates a new version."""
    with temp_db.transaction_cursor() as cursor:
        conversation = Conversation.create(cursor)

        # Write first version
        block1 = Block.create_tool_use(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            name="WritePrompt",
            use_id="write_v1",
            params=(
                '{"key": "versioned", '
                '"summary": "Version 1", '
                '"template": "First version: $var"}'
            ),
        )

        tool1 = WritePrompt(
            tool_id="write_v1",
            tool_name="WritePrompt",
            key="versioned",
            summary="Version 1",
            template="First version: $var",
        )

        tool1.apply(block1)

        # Small delay for timestamp difference
        time.sleep(0.001)

        # Write second version with same key
        block2 = Block.create_tool_use(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            name="WritePrompt",
            use_id="write_v2",
            params=(
                '{"key": "versioned", '
                '"summary": "Version 2", '
                '"template": "Second version: $var with $extra"}'
            ),
        )

        tool2 = WritePrompt(
            tool_id="write_v2",
            tool_name="WritePrompt",
            key="versioned",
            summary="Version 2",
            template="Second version: $var with $extra",
        )

        tool2.apply(block2)

        # Both should succeed
        updated_block1 = Block.get_by_id(cursor, block1.id)
        updated_block2 = Block.get_by_id(cursor, block2.id)
        assert updated_block1 is not None
        assert updated_block2 is not None
        assert updated_block1.tool_response == "Prompt stored"
        assert updated_block2.tool_response == "Prompt stored"

        # Verify latest version is returned
        latest_prompt = Prompt.get_prompt(cursor, "versioned")
        assert latest_prompt is not None
        assert latest_prompt.summary == "Version 2"
        assert latest_prompt.template.template == "Second version: $var with $extra"
