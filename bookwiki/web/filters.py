from datetime import datetime, timezone
from typing import Any, Optional

import markdown  # type: ignore
from flask import Flask
from markupsafe import Markup

from bookwiki.models.block import Block
from bookwiki.models.chapter import Chapter
from bookwiki.models.conversation import Conversation
from bookwiki.tools import deserialize_tool, get_all_tools
from bookwiki.tools.chapter import ReadChapter
from bookwiki.tools.prompt import ShowPrompt, WritePrompt
from bookwiki.tools.system import SpawnAgent
from bookwiki.tools.wiki import ReadWikiPage, SearchWikiByName, WriteWikiPage
from bookwiki.utils import extract_wiki_links


def register_filters(app: Flask) -> None:
    app.jinja_env.filters["markdown"] = markdown_filter
    app.jinja_env.filters["markdown_with_wiki_links"] = markdown_with_wiki_links
    app.jinja_env.filters["extract_block_links"] = extract_block_links
    app.jinja_env.filters["format_chapter_title"] = format_chapter_title
    app.jinja_env.filters["get_conversation_prompt_key"] = get_conversation_prompt_key
    app.jinja_env.filters["format_local_datetime"] = format_local_datetime
    app.jinja_env.finalize = comma_int


def comma_int(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return f"{value:,}"
    return value


def markdown_with_wiki_links(text: Optional[str], chapter_id: int) -> Optional[str]:
    """Convert Markdown to HTML with wiki links transformed to chapter-aware URLs.

    Args:
        text: The Markdown text to convert
        chapter_id: Current chapter for generating wiki URLs

    Returns:
        HTML with wiki links transformed to /wiki/{chapter_id}/{slug} format
    """
    if not text:
        return text

    # Extract all wiki links using the shared utility
    wiki_links = extract_wiki_links(text)

    # Transform text by replacing each original link with chapter-aware version
    transformed_text = text
    for link in wiki_links:
        original_link = f"[{link.display_text}]({link.target})"
        new_link = f"[{link.display_text}](/wiki/{chapter_id}/{link.slug})"
        transformed_text = transformed_text.replace(original_link, new_link)

    # Convert to HTML using markdown
    html_content = markdown.markdown(transformed_text)

    return Markup(html_content)


def markdown_filter(text: Optional[str]) -> Optional[str]:
    """Convert Markdown to HTML without link transformation.

    Args:
        text: The Markdown text to convert

    Returns:
        HTML content with original links preserved
    """
    if not text:
        return text

    # Convert to HTML using markdown without any link transformation
    html_content = markdown.markdown(text)

    return Markup(html_content)


def format_chapter_title(chapter: Chapter) -> str:
    """Format a chapter title from its name list.

    Args:
        chapter: The chapter to format a title for

    Returns:
        Formatted title: " - ".join(chapter.name) if name exists, else "Chapter {id}"
    """
    if chapter.name:
        return " - ".join(chapter.name)
    return f"Chapter {chapter.id}"


def get_conversation_prompt_key(conversation: Conversation) -> Optional[str]:
    """Extract the prompt_key from a conversation's parent block tool parameters.

    Args:
        conversation: The conversation to extract prompt_key from

    Returns:
        The prompt_key string if found, or None if not available
    """
    if not conversation.parent_block:
        return None

    tool_params_json = conversation.parent_block.tool_params_json
    if not tool_params_json:
        return None

    return tool_params_json.get("prompt_key")


def extract_block_links(block: Block) -> list[tuple[str, str]]:
    """Extract all possible links from a block.

    Args:
        block: The block to extract links from
        cursor: Optional database cursor for additional queries

    Returns:
        List of (name, url) tuples for all extractable links
    """

    links: list[tuple[str, str]] = []

    # Handle tool use blocks
    if block.tool_name is not None and block.tool_params is not None:
        try:
            tool = deserialize_tool(
                block.tool_params,
                get_all_tools(),
                tool_id=block.tool_use_id or "",
                tool_name=block.tool_name,
            )
        except Exception:
            # If deserialization fails, skip link extraction for this block
            return links

        if isinstance(tool, ReadChapter):
            # Get the chapter associated with this block's conversation
            conversation = block.conversation
            chapter = conversation.chapter
            links.append(("chapter", f"/chapter/{chapter.id}"))

        elif isinstance(tool, ReadWikiPage):
            # This is a bit sloppy in that it links to the latest version from
            # this chapter, not to the specific version that was read by the llm (but if
            # you want that, just look at the response)
            chapter_id = block.conversation.chapter.id
            links.append(("wiki page", f"/wiki/{chapter_id}/{tool.slug}"))

        elif isinstance(tool, WriteWikiPage):
            wiki_page = block.created_wiki_page
            if wiki_page:
                links.append(
                    ("wiki page", f"/wiki/{wiki_page.chapter_id}/{wiki_page.slug}")
                )

        elif isinstance(tool, SearchWikiByName):
            # Get the chapter associated with this block's conversation
            conversation = block.conversation
            chapter = conversation.chapter
            names_str = ",".join(tool.names)
            links.append(("search", f"/search/{chapter.id}?names={names_str}"))

        elif isinstance(tool, WritePrompt):
            # For WritePrompt, this block created the prompt, so use this block's ID
            links.append(("prompt", f"/prompt/{tool.key}#block-{block.id}"))

        elif isinstance(tool, ShowPrompt):
            # For ShowPrompt, just show the latest prompt version
            links.append(("prompt", f"/prompt/{tool.key}"))

        elif isinstance(tool, SpawnAgent):
            # For SpawnAgent, just show the latest prompt version
            links.append(("prompt", f"/prompt/{tool.prompt_key}"))
            spawned_conversation = block.spawned_conversation
            if spawned_conversation:
                links.append(
                    ("conversation", f"/conversation/{spawned_conversation.id}")
                )

    return links


def format_local_datetime(dt: Optional[datetime]) -> str:
    """Format a datetime object to local time in YYYY-MM-DD HH:MM:SS format.

    Args:
        dt: The datetime object to format (assumed to be in UTC)

    Returns:
        Formatted datetime string in local time, or empty string if None
    """
    if not dt:
        return ""

    # Ensure the datetime has UTC timezone info
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    # Convert to local time
    local_dt = dt.astimezone()

    # Format without timezone info
    return local_dt.strftime("%Y-%m-%d %H:%M:%S")
