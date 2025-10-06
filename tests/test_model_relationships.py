"""Multi-model relationship tests for bookwiki."""

from string import Template

from bookwiki.db import SafeConnection
from bookwiki.models import Chapter, Conversation, Prompt, WikiPage


def test_block_to_conversation_to_wiki_page_chain(temp_db: SafeConnection) -> None:
    """Test the complete chain: Conversation -> Block -> WikiPage -> Chapter."""
    with temp_db.transaction_cursor() as cursor:
        # Set up chapter
        chapter = Chapter.add_chapter(
            cursor,
            chapter_id=1,
            name=["The Hobbit", "Chapter 1"],
            text="In a hole in the ground...",
        )

        # Create conversation
        conversation = Conversation.create(cursor)
        conversation.update_tokens(100, 50)

        # Create block in conversation
        block = conversation.add_assistant_text("I'll create a wiki page")

        # Create wiki page from block
        wiki_page = block.write_wiki_page(
            chapter_id=chapter.id,
            slug="bilbo-baggins",
            title="Bilbo Baggins",
            names=["Bilbo", "Mr. Baggins"],
            summary="A hobbit from the Shire",
            body="Bilbo is the main character of The Hobbit.",
        )

        # Verify all relationships
        assert wiki_page.create_block_id == block.id
        assert wiki_page.chapter_id == chapter.id
        assert block.conversation_id == conversation.id

        # Test reverse lookups
        from bookwiki.models import Block

        # From wiki page, get creating block
        create_block = Block.get_by_id(cursor, wiki_page.create_block_id)
        assert create_block is not None
        assert create_block.conversation_id == conversation.id

        # From block, get conversation
        block_conversation = Conversation.get_by_id(cursor, block.conversation_id)
        assert block_conversation is not None
        assert block_conversation.total_input_tokens == 100

        # From wiki page, get chapter
        wiki_chapter = Chapter.read_chapter(cursor, wiki_page.chapter_id)
        assert wiki_chapter is not None
        assert "hole in the ground" in wiki_chapter.text


def test_prompt_creation_and_retrieval_patterns(temp_db: SafeConnection) -> None:
    """Test prompt creation, versioning, and retrieval patterns."""
    with temp_db.transaction_cursor() as cursor:
        # Create conversation and blocks
        conversation = Conversation.create(cursor)
        block1 = conversation.add_assistant_text("Creating first prompt")
        block2 = conversation.add_assistant_text("Creating second version")
        block3 = conversation.add_assistant_text("Creating different prompt")

        # Create initial prompt
        template1 = Template("Hello $name, please $action")
        block1.add_prompt(
            key="greeting",
            summary="Basic greeting prompt",
            template=template1,
        )

        # Create updated version of same prompt
        template2 = Template("Hi $name, could you please $action?")
        block2.add_prompt(
            key="greeting",
            summary="Polite greeting prompt",
            template=template2,
        )

        # Create different prompt
        template3 = Template("Analyze $content for $purpose")
        block3.add_prompt(
            key="analysis",
            summary="Content analysis prompt",
            template=template3,
        )

        # Test get_prompt returns latest version
        latest_greeting = Prompt.get_prompt(cursor, "greeting")
        assert latest_greeting is not None
        assert latest_greeting.summary == "Polite greeting prompt"
        assert latest_greeting.create_block_id == block2.id

        # Test get_prompt for different key
        analysis_prompt = Prompt.get_prompt(cursor, "analysis")
        assert analysis_prompt is not None
        assert analysis_prompt.summary == "Content analysis prompt"
        assert analysis_prompt.create_block_id == block3.id

        # Test list_prompts returns latest versions only
        all_prompts = Prompt.list_prompts(cursor)
        assert len(all_prompts) == 2
        assert "greeting" in all_prompts
        assert "analysis" in all_prompts
        assert all_prompts["greeting"].summary == "Polite greeting prompt"

        # Test create_block method
        create_block = latest_greeting.create_block
        assert create_block.id == block2.id
        assert create_block.text_body == "Creating second version"


def test_wiki_page_name_relationships(temp_db: SafeConnection) -> None:
    """Test complex wiki page name relationships and lookups."""
    with temp_db.transaction_cursor() as cursor:
        # Set up data
        Chapter.add_chapter(cursor, 1, ["Book"], "Content")
        conversation = Conversation.create(cursor)

        # Create wiki pages with overlapping names
        block1 = conversation.add_assistant_text("Creating Gandalf page")
        gandalf_page = block1.write_wiki_page(
            chapter_id=1,
            slug="gandalf",
            title="Gandalf the Grey",
            names=["Gandalf", "Gandalf the Grey", "The Grey Wizard"],
            summary="A wizard",
            body="Gandalf is a wizard.",
        )

        block2 = conversation.add_assistant_text("Creating Saruman page")
        saruman_page = block2.write_wiki_page(
            chapter_id=1,
            slug="saruman",
            title="Saruman the White",
            names=["Saruman", "Saruman the White", "The White Wizard"],
            summary="Another wizard",
            body="Saruman is also a wizard.",
        )

        block3 = conversation.add_assistant_text("Creating generic wizard page")
        wizard_page = block3.write_wiki_page(
            chapter_id=1,
            slug="wizards",
            title="Wizards of Middle-earth",
            names=["Wizards", "The Istari"],
            summary="Magical beings",
            body="Wizards are magical beings in Middle-earth.",
        )

        # Test name retrieval
        assert set(gandalf_page.names) == {
            "Gandalf",
            "Gandalf the Grey",
            "The Grey Wizard",
        }
        assert set(saruman_page.names) == {
            "Saruman",
            "Saruman the White",
            "The White Wizard",
        }
        assert set(wizard_page.names) == {"Wizards", "The Istari"}

        # Test that names are properly normalized by verifying all expected names
        # are accessible through the model layer
        name_slug_pairs = WikiPage.get_name_slug_pairs(cursor, 1)
        all_names = {name for name, slug in name_slug_pairs}
        expected_names = {
            "Gandalf",
            "Gandalf the Grey",
            "The Grey Wizard",
            "Saruman",
            "Saruman the White",
            "The White Wizard",
            "Wizards",
            "The Istari",
        }
        assert all_names == expected_names

        # Test name reuse doesn't create duplicates by verifying that reusing
        # "Gandalf" doesn't affect name accessibility
        block4 = conversation.add_assistant_text("Creating another Gandalf reference")
        block4.write_wiki_page(
            chapter_id=1,
            slug="grey-pilgrim",
            title="The Grey Pilgrim",
            names=["Gandalf", "Grey Pilgrim"],  # "Gandalf" already exists
            summary="Another name for Gandalf",
            body="The Grey Pilgrim is another name for Gandalf.",
        )

        # Verify all names are still accessible plus the new one
        updated_name_slug_pairs = WikiPage.get_name_slug_pairs(cursor, 1)
        updated_all_names = {name for name, slug in updated_name_slug_pairs}
        expected_names.add("Grey Pilgrim")  # Add the new name
        assert updated_all_names == expected_names


def test_conversation_hierarchy_with_complex_operations(
    temp_db: SafeConnection,
) -> None:
    """Test complex hierarchical conversation operations."""
    with temp_db.transaction_cursor() as cursor:
        Chapter.add_chapter(cursor, 1, ["Book"], "Chapter content")

        # Create parent conversation
        parent_conv = Conversation.create(cursor)
        parent_conv.add_user_text("I need analysis of characters and plot")

        # Parent creates specialized agents
        parent_conv.add_assistant_text("Spawning character and plot agents")
        char_spawn_block = parent_conv.add_tool_use(
            name="SpawnAgent",
            use_id="spawn_char",
            params='{"prompt_key": "char_analysis", "template_vars": {}}',
        )

        # Create character analysis child
        char_conv = char_spawn_block.start_conversation()
        char_conv.add_user_text("Analyze main characters")
        char_response_block = char_conv.add_assistant_text(
            "Creating character wiki pages"
        )

        # Character agent creates wiki page
        char_wiki = char_response_block.write_wiki_page(
            chapter_id=1,
            slug="frodo",
            title="Frodo Baggins",
            names=["Frodo"],
            summary="Main protagonist",
            body="Frodo is the ring bearer.",
        )

        # Character agent creates prompt for future use
        char_response_block.add_prompt(
            key="character-analysis",
            summary="Template for analyzing characters",
            template=Template("Analyze character $name from chapter $chapter"),
        )

        # Create plot analysis child
        plot_spawn_block = parent_conv.add_tool_use(
            name="SpawnAgent",
            use_id="spawn_plot",
            params='{"prompt_key": "plot_analysis", "template_vars": {}}',
        )
        plot_conv = plot_spawn_block.start_conversation()
        plot_conv.add_user_text("Analyze plot structure")
        plot_response_block = plot_conv.add_assistant_text("Analyzing plot progression")

        # Plot agent creates different wiki page
        plot_wiki = plot_response_block.write_wiki_page(
            chapter_id=1,
            slug="the-quest",
            title="The Quest",
            names=["The Quest", "The Journey"],
            summary="The main plot line",
            body="The quest to destroy the ring.",
        )

        # Verify hierarchical relationships
        assert char_conv.parent_block_id == char_spawn_block.id
        assert plot_conv.parent_block_id == plot_spawn_block.id
        assert char_spawn_block.conversation_id == parent_conv.id
        assert plot_spawn_block.conversation_id == parent_conv.id

        # Verify each child has created distinct resources
        assert char_wiki.slug != plot_wiki.slug
        assert char_wiki.create_block_id == char_response_block.id
        assert plot_wiki.create_block_id == plot_response_block.id

        # Verify prompt was created and can be retrieved
        retrieved_prompt = Prompt.get_prompt(cursor, "character-analysis")
        assert retrieved_prompt is not None
        assert retrieved_prompt.create_block_id == char_response_block.id

        # Verify create_block works across the hierarchy
        prompt_creator = retrieved_prompt.create_block
        assert prompt_creator.id == char_response_block.id
        assert prompt_creator.conversation_id == char_conv.id


def test_orphaned_entity_cleanup_patterns(temp_db: SafeConnection) -> None:
    """Test patterns for identifying potentially orphaned entities."""
    with temp_db.transaction_cursor() as cursor:
        Chapter.add_chapter(cursor, 1, ["Book"], "Content")

        # Create a conversation with various entities
        conversation = Conversation.create(cursor)

        # Create blocks
        conversation.add_user_text("Create content")
        assistant_block = conversation.add_assistant_text("Creating wiki and prompt")

        # Create wiki page
        assistant_block.write_wiki_page(
            chapter_id=1,
            slug="test-entity",
            title="Test Entity",
            names=["Entity", "Test"],
            summary="Test summary",
            body="Test body",
        )

        # Create prompt
        assistant_block.add_prompt(
            key="test-prompt",
            summary="Test prompt",
            template=Template("Test template with $var"),
        )

        # Verify everything is properly linked
        # Wiki page -> Block -> Conversation
        from bookwiki.models import WikiPage

        wiki_page = WikiPage.read_page_at(cursor, "test-entity", 1)
        assert wiki_page is not None
        assert wiki_page.create_block.conversation_id == conversation.id

        # Prompt -> Block -> Conversation
        test_prompt = Prompt.get_prompt(cursor, "test-prompt")
        assert test_prompt is not None
        assert test_prompt.create_block.conversation_id == conversation.id

        # Test finding entities created by specific blocks
        wiki_page = assistant_block.created_wiki_page
        assert wiki_page is not None

        # Verify prompt was created by this block through the get_prompt call above
        assert test_prompt.create_block_id == assistant_block.id

        # Test finding entities in specific conversations
        # Verify the wiki page was created within this conversation
        assert wiki_page.create_block.conversation_id == conversation.id


def test_complex_wiki_name_sharing(temp_db: SafeConnection) -> None:
    """Test complex scenarios with shared wiki names across pages."""
    with temp_db.transaction_cursor() as cursor:
        Chapter.add_chapter(cursor, 1, ["Book", "Ch1"], "Content 1")
        Chapter.add_chapter(cursor, 2, ["Book", "Ch2"], "Content 2")

        conversation = Conversation.create(cursor)

        # Create multiple pages that share some names
        block1 = conversation.add_assistant_text("Creating Aragorn page")
        block1.write_wiki_page(
            chapter_id=1,
            slug="aragorn",
            title="Strider",
            names=["Aragorn", "Strider", "Ranger"],
            summary="A mysterious ranger",
            body="A ranger of the North.",
        )

        block2 = conversation.add_assistant_text("Updating Aragorn page")
        block2.write_wiki_page(
            chapter_id=2,
            slug="aragorn",
            title="Aragorn, son of Arathorn",
            names=["Aragorn", "King Elessar", "Heir of Isildur"],
            summary="Rightful king of Gondor",
            body="Revealed as the true king.",
        )

        block3 = conversation.add_assistant_text("Creating Rangers page")
        block3.write_wiki_page(
            chapter_id=1,
            slug="rangers",
            title="Rangers of the North",
            names=["Rangers", "Dúnedain"],
            summary="Protectors of the North",
            body="The Rangers protect the northern lands.",
        )

        # Verify that name deduplication works by checking that the same name
        # can be used across different versions of the same page
        aragorn_v1 = WikiPage.read_page_at(cursor, "aragorn", 1)
        aragorn_v2 = WikiPage.read_page_at(cursor, "aragorn", 2)
        assert aragorn_v1 is not None
        assert aragorn_v2 is not None
        # Both versions should have "Aragorn" name - this tests that the name
        # is properly shared/deduplicated in the database
        assert "Aragorn" in aragorn_v1.names
        assert "Aragorn" in aragorn_v2.names

        # But different pages reference it
        # Count distinct current wiki pages that have the name "Aragorn"
        name_slug_pairs = WikiPage.get_name_slug_pairs(cursor, 2)
        aragorn_slugs = {slug for name, slug in name_slug_pairs if name == "Aragorn"}
        assert len(aragorn_slugs) == 1  # One current page references "Aragorn"

        # Verify complex name queries work
        # Find all pages that have ever had "Rangers" or "Ranger" names
        # This tests historical name queries across all wiki content, not just current
        target_names = ["Ranger", "Rangers"]
        all_ranger_pages = set()

        # Check all wiki pages that have ever existed for these names
        # We need to check all chapters to find where these names appeared
        for chapter_num in [1, 2]:
            chapter_name_pairs = WikiPage.get_name_slug_pairs(cursor, chapter_num)
            for name, slug in chapter_name_pairs:
                if name in target_names:
                    all_ranger_pages.add(slug)

        # Also check any pages that might not be visible at specific chapters
        # by checking versions of known slugs
        for slug in ["aragorn", "rangers"]:
            versions = WikiPage.get_versions_by_slug(cursor, slug, 2)
            for version in versions:
                if any(name in target_names for name in version.names):
                    all_ranger_pages.add(slug)

        assert all_ranger_pages == {"aragorn", "rangers"}
