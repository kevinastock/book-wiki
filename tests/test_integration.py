"""Integration tests for bookwiki complex conversation workflows."""

from bookwiki.db import SafeConnection
from bookwiki.models import Chapter, Conversation, WikiPage
from bookwiki.tools.chapter import ReadChapter
from bookwiki.tools.wiki import WriteWikiPage


def test_conversation_with_multiple_tool_uses_and_wiki_creation(
    temp_db: SafeConnection,
) -> None:
    """Test a complete workflow: read chapter, create wiki page, update it."""
    with temp_db.transaction_cursor() as cursor:
        # Set up chapter data
        ch = Chapter.add_chapter(
            cursor,
            chapter_id=1,
            name=["The Hobbit", "Chapter 1", "An Unexpected Party"],
            text="In a hole in the ground there lived a hobbit...",
        )

        # Start a conversation
        conversation = Conversation.create(cursor)
        # Mark chapter as started by this conversation
        ch.set_conversation_id(conversation)

        # User requests to read chapter and create wiki page
        conversation.add_user_text(
            "Please read chapter 1 and create a wiki page for Bilbo Baggins"
        )

        # Assistant responds and uses tools
        conversation.add_assistant_text(
            "I'll read the chapter and create a wiki page for Bilbo Baggins."
        )

        # Tool use: Read chapter
        read_tool_block = conversation.add_tool_use(
            "ReadChapter", "read_ch1", '{"chapter_offset": 0}'
        )

        # Apply the ReadChapter tool
        read_tool = ReadChapter(
            tool_id="read_ch1", tool_name="ReadChapter", chapter_offset=0
        )
        read_tool.apply(read_tool_block)

        # Tool use: Create wiki page
        write_tool_block = conversation.add_tool_use(
            "WriteWikiPage",
            "create_bilbo",
            '{"slug": "bilbo-baggins", "title": "Bilbo Baggins", '
            '"names": ["Bilbo Baggins", "Bilbo"], '
            '"summary": "A hobbit from the Shire", '
            '"body": "Bilbo Baggins is the protagonist hobbit.", "create": true}',
        )

        # Apply the WriteWikiPage tool
        write_tool = WriteWikiPage(
            tool_id="create_bilbo",
            tool_name="WriteWikiPage",
            slug="bilbo-baggins",
            title="Bilbo Baggins",
            names=["Bilbo Baggins", "Bilbo"],
            summary="A hobbit from the Shire",
            body="Bilbo Baggins is the protagonist hobbit.",
            create=True,
        )
        write_tool.apply(write_tool_block)

        # Verify conversation is now ready (all tools completed)
        ready_conversation = Conversation.find_sendable_conversation(cursor)
        assert ready_conversation is not None
        assert ready_conversation.id == conversation.id

        # Verify all blocks are present and in correct order
        all_blocks = conversation.blocks
        assert len(all_blocks) == 4

        assert all_blocks[0].text_role == "user"
        assert all_blocks[1].text_role == "assistant"
        assert all_blocks[2].tool_name == "ReadChapter"
        assert all_blocks[2].tool_response is not None
        assert "An Unexpected Party" in all_blocks[2].tool_response
        assert all_blocks[3].tool_name == "WriteWikiPage"
        assert all_blocks[3].tool_response == "Wrote wiki page"

        # Verify wiki page was actually created
        wiki_page = WikiPage.read_page_at(cursor, "bilbo-baggins", 1)
        assert wiki_page is not None
        assert wiki_page.title == "Bilbo Baggins"
        assert set(wiki_page.names) == {"Bilbo Baggins", "Bilbo"}


def test_parent_child_conversation_relationships(temp_db: SafeConnection) -> None:
    """Test spawning child conversations and their relationships."""
    with temp_db.transaction_cursor() as cursor:
        # Create parent conversation
        parent_conversation = Conversation.create(cursor)

        # Add some activity to parent
        parent_conversation.add_user_text("I need help with multiple tasks")
        parent_conversation.add_assistant_text(
            "I'll spawn specialized agents for each task"
        )
        spawn_block = parent_conversation.add_tool_use(
            name="SpawnAgent",
            use_id="spawn_1",
            params='{"prompt_key": "test", "template_vars": {}}',
        )

        # Spawn first child conversation
        child1_conversation = spawn_block.start_conversation()
        assert child1_conversation.parent_block_id == spawn_block.id
        assert child1_conversation.id != parent_conversation.id

        # Create another tool use block for the second spawn
        spawn_block2 = parent_conversation.add_tool_use(
            name="SpawnAgent",
            use_id="spawn_2",
            params='{"prompt_key": "test", "template_vars": {}}',
        )

        # Spawn second child conversation
        child2_conversation = spawn_block2.start_conversation()
        assert child2_conversation.parent_block_id == spawn_block2.id
        assert child2_conversation.id != parent_conversation.id
        assert child2_conversation.id != child1_conversation.id

        # Add activity to child conversations
        child1_conversation.add_user_text("Task 1: Create character wiki")
        child2_conversation.add_user_text("Task 2: Analyze plot structure")

        # Verify parent-child relationships
        assert child1_conversation.parent_block_id == spawn_block.id
        assert child2_conversation.parent_block_id == spawn_block2.id

        # Verify each conversation has its own blocks
        parent_block_ids = parent_conversation.blocks
        child1_blocks = child1_conversation.blocks
        child2_blocks = child2_conversation.blocks

        assert (
            len(parent_block_ids) == 4
        )  # user text, assistant text, spawn_1 tool, spawn_2 tool
        assert len(child1_blocks) == 1
        assert len(child2_blocks) == 1

        # Verify block IDs don't overlap
        all_block_ids = (
            [b.id for b in parent_block_ids]
            + [b.id for b in child1_blocks]
            + [b.id for b in child2_blocks]
        )
        assert len(all_block_ids) == len(set(all_block_ids))  # All unique


def test_conversation_token_tracking_across_multiple_interactions(
    temp_db: SafeConnection,
) -> None:
    """Test token counting accumulation across multiple LLM interactions."""
    with temp_db.transaction_cursor() as cursor:
        conversation = Conversation.create(cursor)

        # Initial state - zero tokens
        assert conversation.total_input_tokens == 0
        assert conversation.total_output_tokens == 0
        assert conversation.current_tokens == 0

        # First interaction
        conversation.update_tokens(100, 75)

        # Verify database was updated
        updated_conversation = Conversation.get_by_id(cursor, conversation.id)
        assert updated_conversation is not None
        assert updated_conversation.total_input_tokens == 100
        assert updated_conversation.total_output_tokens == 75
        assert updated_conversation.current_tokens == 175

        # Second interaction
        conversation.update_tokens(50, 60)

        updated_conversation = Conversation.get_by_id(cursor, conversation.id)
        assert updated_conversation is not None
        assert updated_conversation.total_input_tokens == 150  # 100 + 50
        assert updated_conversation.total_output_tokens == 135  # 75 + 60
        assert updated_conversation.current_tokens == 110  # 50 + 60 (latest only)

        # Third interaction
        conversation.update_tokens(25, 30)

        updated_conversation = Conversation.get_by_id(cursor, conversation.id)
        assert updated_conversation is not None
        assert updated_conversation.total_input_tokens == 175  # 150 + 25
        assert updated_conversation.total_output_tokens == 165  # 135 + 30
        assert updated_conversation.current_tokens == 55  # 25 + 30 (latest only)


def test_complex_conversation_state_transitions(temp_db: SafeConnection) -> None:
    """Test complex patterns of conversation readiness state transitions."""
    with temp_db.transaction_cursor() as cursor:
        Chapter.add_chapter(cursor, 1, ["Book", "Chapter 1"], "Content...")

        conversation = Conversation.create(cursor)

        # Initially not ready (no unsent blocks)
        ready = Conversation.find_sendable_conversation(cursor)
        assert ready is None

        # Add user message - now ready
        conversation.add_user_text("Please help")
        ready = Conversation.find_sendable_conversation(cursor)
        assert ready is not None
        assert ready.id == conversation.id

        # Add incomplete tool use - no longer ready
        tool_block1 = conversation.add_tool_use(
            "ReadChapter", "read1", '{"chapter_offset": 0}'
        )
        ready = Conversation.find_sendable_conversation(cursor)
        assert ready is None

        # Add another incomplete tool use - still not ready
        tool_block2 = conversation.add_tool_use(
            "ReadChapter", "read2", '{"chapter_offset": 0}'
        )
        ready = Conversation.find_sendable_conversation(cursor)
        assert ready is None

        # Complete first tool - still not ready (second incomplete)
        tool_block1.respond("Chapter 1 content")
        ready = Conversation.find_sendable_conversation(cursor)
        assert ready is None

        # Complete second tool - now ready again
        tool_block2.respond("Chapter 1 content again")
        ready = Conversation.find_sendable_conversation(cursor)
        assert ready is not None
        assert ready.id == conversation.id

        # Add assistant response and mark user message as sent
        # Note: conversation remains ready because tool responses are complete
        conversation.add_assistant_text("Here's the information you requested")
        all_blocks = conversation.blocks
        user_block = next(b for b in all_blocks if b.text_role == "user")
        user_block.mark_as_sent()

        ready = Conversation.find_sendable_conversation(cursor)
        # Still ready because assistant message and completed tools make it ready
        assert ready is not None
        assert ready.id == conversation.id


def test_wiki_page_versioning_across_chapters(temp_db: SafeConnection) -> None:
    """Test wiki page evolution across different chapters."""
    with temp_db.transaction_cursor() as cursor:
        # Set up chapters
        Chapter.add_chapter(cursor, 1, ["Book", "Chapter 1"], "Bilbo introduced")
        Chapter.add_chapter(cursor, 5, ["Book", "Chapter 5"], "Bilbo finds ring")
        Chapter.add_chapter(cursor, 10, ["Book", "Chapter 10"], "Bilbo uses ring")

        conversation = Conversation.create(cursor)

        # Create initial wiki page at chapter 1
        create_block_ch1 = conversation.add_assistant_text("Creating initial page")
        create_block_ch1.write_wiki_page(
            chapter_id=1,
            slug="bilbo-baggins",
            title="Bilbo Baggins",
            names=["Bilbo"],
            summary="A hobbit from the Shire",
            body="Bilbo is introduced as a respectable hobbit.",
        )

        # Update wiki page at chapter 5
        create_block_ch5 = conversation.add_assistant_text("Updating page")
        create_block_ch5.write_wiki_page(
            chapter_id=5,
            slug="bilbo-baggins",
            title="Bilbo Baggins",
            names=["Bilbo", "The Burglar"],
            summary="A hobbit adventurer with a magic ring",
            body="Bilbo found a magic ring during his adventure.",
        )

        # Update wiki page at chapter 10
        create_block_ch10 = conversation.add_assistant_text("Final update")
        create_block_ch10.write_wiki_page(
            chapter_id=10,
            slug="bilbo-baggins",
            title="Bilbo Baggins",
            names=["Bilbo", "The Invisible Burglar"],
            summary="A hobbit master of the One Ring",
            body="Bilbo has mastered the use of the invisible ring.",
        )

        # Verify different versions at different chapters
        page_at_ch1 = WikiPage.read_page_at(cursor, "bilbo-baggins", 1)
        page_at_ch5 = WikiPage.read_page_at(cursor, "bilbo-baggins", 5)
        page_at_ch10 = WikiPage.read_page_at(cursor, "bilbo-baggins", 10)

        assert page_at_ch1 is not None
        assert page_at_ch5 is not None
        assert page_at_ch10 is not None

        # Verify chapter 1 version
        assert page_at_ch1.names == ["Bilbo"]
        assert "respectable hobbit" in page_at_ch1.body

        # Verify chapter 5 version
        assert set(page_at_ch5.names) == {"Bilbo", "The Burglar"}
        assert "magic ring" in page_at_ch5.body

        # Verify chapter 10 version
        assert set(page_at_ch10.names) == {"Bilbo", "The Invisible Burglar"}
        assert "mastered" in page_at_ch10.body

        # Verify they're different entities
        assert page_at_ch1.id != page_at_ch5.id != page_at_ch10.id
        assert (
            page_at_ch1.create_time
            != page_at_ch5.create_time
            != page_at_ch10.create_time
        )
