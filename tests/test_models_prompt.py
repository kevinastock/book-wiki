"""Tests for bookwiki Prompt model."""

import time
from string import Template

from bookwiki.db import SafeConnection
from bookwiki.models import Block, Conversation, Prompt


def test_prompt_write_and_get(temp_db: SafeConnection) -> None:
    """Test writing and retrieving a prompt."""
    with temp_db.transaction_cursor() as cursor:
        # Create conversation and block first
        conversation = Conversation.create(cursor)
        block = Block.create_text(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            role="assistant",
            text="Creating a prompt",
        )

        # Create a prompt
        template_str = (
            "Please analyze chapter $chapter_num and create a wiki page for "
            "$character_name"
        )
        template = Template(template_str)

        # Create and write the prompt to database
        Prompt.create(
            cursor=cursor,
            key="analyze_character",
            create_block_id=block.id,
            summary="Analyzes a character from a specific chapter",
            template=template,
        )

        # Retrieve it back
        retrieved_prompt = Prompt.get_prompt(cursor, "analyze_character")

        assert retrieved_prompt is not None
        assert retrieved_prompt.key == "analyze_character"
        assert (
            retrieved_prompt.summary == "Analyzes a character from a specific chapter"
        )
        assert retrieved_prompt.template.template == template_str
        assert retrieved_prompt.create_block_id == block.id


def test_prompt_get_nonexistent(temp_db: SafeConnection) -> None:
    """Test retrieving a prompt that doesn't exist."""
    with temp_db.transaction_cursor() as cursor:
        prompt = Prompt.get_prompt(cursor, "nonexistent_key")
        assert prompt is None


def test_prompt_versioning(temp_db: SafeConnection) -> None:
    """Test that prompt versioning works (latest version is returned)."""
    with temp_db.transaction_cursor() as cursor:
        # Create conversation and blocks
        conversation = Conversation.create(cursor)
        block1 = Block.create_text(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            role="assistant",
            text="Creating first version",
        )
        block2 = Block.create_text(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            role="assistant",
            text="Creating second version",
        )

        # Create first version of prompt
        template1 = Template("Version 1: Analyze $character")
        Prompt.create(
            cursor=cursor,
            key="versioned_prompt",
            create_block_id=block1.id,
            summary="First version",
            template=template1,
        )

        # Wait a tiny bit to ensure different timestamps
        time.sleep(0.001)

        # Create second version of prompt (same key, different content)
        template2 = Template(
            "Version 2: Deeply analyze $character from chapter $chapter"
        )
        Prompt.create(
            cursor=cursor,
            key="versioned_prompt",
            create_block_id=block2.id,
            summary="Second version with more detail",
            template=template2,
        )

        # Get prompt should return the latest version
        retrieved_prompt = Prompt.get_prompt(cursor, "versioned_prompt")

        assert retrieved_prompt is not None
        assert retrieved_prompt.key == "versioned_prompt"
        assert retrieved_prompt.summary == "Second version with more detail"
        assert (
            retrieved_prompt.template.template
            == "Version 2: Deeply analyze $character from chapter $chapter"
        )
        assert retrieved_prompt.create_block_id == block2.id


def test_prompt_list_prompts_empty(temp_db: SafeConnection) -> None:
    """Test listing prompts when none exist."""
    with temp_db.transaction_cursor() as cursor:
        prompts = Prompt.list_prompts(cursor)
        assert prompts == {}


def test_prompt_list_prompts_multiple(temp_db: SafeConnection) -> None:
    """Test listing multiple prompts."""
    with temp_db.transaction_cursor() as cursor:
        conversation = Conversation.create(cursor)

        # Create blocks for prompts
        block1 = Block.create_text(
            cursor,
            conversation.id,
            conversation.current_generation,
            "assistant",
            "Block 1",
        )
        block2 = Block.create_text(
            cursor,
            conversation.id,
            conversation.current_generation,
            "assistant",
            "Block 2",
        )
        block3 = Block.create_text(
            cursor,
            conversation.id,
            conversation.current_generation,
            "assistant",
            "Block 3",
        )

        # Create multiple prompts
        prompts_data = [
            (
                "character_analyzer",
                "Analyzes character traits",
                "Analyze $character in chapter $chapter",
            ),
            (
                "location_describer",
                "Describes locations",
                "Describe the location $location",
            ),
            (
                "plot_summarizer",
                "Summarizes plot points",
                "Summarize events in chapter $chapter",
            ),
        ]

        blocks = [block1, block2, block3]

        for i, (key, summary, template_str) in enumerate(prompts_data):
            Prompt.create(
                cursor=cursor,
                key=key,
                create_block_id=blocks[i].id,
                summary=summary,
                template=Template(template_str),
            )

            # Small delay to ensure different timestamps
            time.sleep(0.001)

        # List all prompts
        all_prompts = Prompt.list_prompts(cursor)

        assert len(all_prompts) == 3
        assert "character_analyzer" in all_prompts
        assert "location_describer" in all_prompts
        assert "plot_summarizer" in all_prompts

        # Verify content
        assert all_prompts["character_analyzer"].summary == "Analyzes character traits"
        assert (
            all_prompts["location_describer"].template.template
            == "Describe the location $location"
        )
        assert all_prompts["plot_summarizer"].create_block_id == block3.id


def test_prompt_list_prompts_with_versions(temp_db: SafeConnection) -> None:
    """Test that list_prompts returns only the latest version of each key."""
    with temp_db.transaction_cursor() as cursor:
        conversation = Conversation.create(cursor)
        block1 = Block.create_text(
            cursor,
            conversation.id,
            conversation.current_generation,
            "assistant",
            "Block 1",
        )
        block2 = Block.create_text(
            cursor,
            conversation.id,
            conversation.current_generation,
            "assistant",
            "Block 2",
        )

        # Create first version
        Prompt.create(
            cursor=cursor,
            key="evolving_prompt",
            create_block_id=block1.id,
            summary="First version",
            template=Template("Old template: $var"),
        )

        time.sleep(0.001)

        # Create second version (should be the one returned)
        Prompt.create(
            cursor=cursor,
            key="evolving_prompt",
            create_block_id=block2.id,
            summary="Updated version",
            template=Template("New template: $var with $extra"),
        )

        # List prompts should only return the latest version
        all_prompts = Prompt.list_prompts(cursor)

        assert len(all_prompts) == 1
        assert "evolving_prompt" in all_prompts

        latest_prompt = all_prompts["evolving_prompt"]
        assert latest_prompt.summary == "Updated version"
        assert latest_prompt.template.template == "New template: $var with $extra"
        assert latest_prompt.create_block_id == block2.id


def test_prompt_create_block(temp_db: SafeConnection) -> None:
    """Test getting the block that created a prompt."""
    with temp_db.transaction_cursor() as cursor:
        conversation = Conversation.create(cursor)
        original_block = Block.create_text(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            role="assistant",
            text="I'm creating a new prompt template",
        )

        # Create a prompt
        prompt = Prompt.create(
            cursor=cursor,
            key="test_prompt",
            create_block_id=original_block.id,
            summary="Test prompt",
            template=Template("Test template with $variable"),
        )

        # Get the create block
        create_block = prompt.create_block

        assert create_block.id == original_block.id
        assert create_block.text_body == "I'm creating a new prompt template"
        assert create_block.conversation_id == conversation.id


def test_prompt_template_substitution(temp_db: SafeConnection) -> None:
    """Test that template substitution works correctly."""
    with temp_db.transaction_cursor() as cursor:
        conversation = Conversation.create(cursor)
        block = Block.create_text(
            cursor,
            conversation.id,
            conversation.current_generation,
            "assistant",
            "Creating template",
        )

        # Create a prompt with multiple variables
        template_str = (
            "Analyze the character $character_name from chapter $chapter_num. "
            "Focus on their $trait and relationship with $other_character."
        )
        Prompt.create(
            cursor=cursor,
            key="complex_template",
            create_block_id=block.id,
            summary="Complex character analysis template",
            template=Template(template_str),
        )

        # Retrieve and test substitution
        retrieved_prompt = Prompt.get_prompt(cursor, "complex_template")
        assert retrieved_prompt is not None

        # Test variable identification
        variables = retrieved_prompt.template.get_identifiers()
        expected_vars = {"character_name", "chapter_num", "trait", "other_character"}
        assert set(variables) == expected_vars

        # Test substitution
        substituted = retrieved_prompt.template.substitute(
            {
                "character_name": "Rand al'Thor",
                "chapter_num": "5",
                "trait": "leadership",
                "other_character": "Egwene",
            }
        )

        expected = (
            "Analyze the character Rand al'Thor from chapter 5. "
            "Focus on their leadership and relationship with Egwene."
        )
        assert substituted == expected


def test_prompt_block_add_prompt_method(temp_db: SafeConnection) -> None:
    """Test the add_prompt method on Block."""
    with temp_db.transaction_cursor() as cursor:
        conversation = Conversation.create(cursor)
        block = Block.create_text(
            cursor=cursor,
            conversation_id=conversation.id,
            generation=conversation.current_generation,
            role="assistant",
            text="I'm creating a prompt via the block method",
        )

        # Use the block's add_prompt method
        template = Template("Block-created template: $variable")
        created_prompt = block.add_prompt(
            key="block_created_prompt",
            summary="Created via block method",
            template=template,
        )

        # Verify the prompt was created
        assert created_prompt.key == "block_created_prompt"
        assert created_prompt.summary == "Created via block method"
        assert created_prompt.create_block_id == block.id
        assert created_prompt.template.template == "Block-created template: $variable"

        # Verify it was written to the database
        retrieved_prompt = Prompt.get_prompt(cursor, "block_created_prompt")
        assert retrieved_prompt is not None
        assert retrieved_prompt.summary == "Created via block method"
