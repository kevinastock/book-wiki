import logging
from sqlite3 import Cursor
from typing import Optional

from bookwiki.models import Block, Chapter, WikiPage
from bookwiki.search import find_similar_slugs, search_wiki_by_name
from bookwiki.tools.base import LLMSolvableError, ToolModel
from bookwiki.utils import extract_wiki_links

logger = logging.getLogger(__name__)


class ReadWikiPage(ToolModel):
    """Retrieve the complete content of a wiki page to review what information is
    already documented.

    Use this tool when you need to check existing wiki content before updating it,
    reference information from other pages, or verify what has already been written
    about a character, location, or concept. This shows the full page including title,
    known names, summary, and body content.
    """

    slug: str
    """The unique identifier for the wiki page (e.g., 'gandalf-the-grey',
    'minas-tirith'). Use lowercase with hyphens between words."""

    def _apply(self, block: Block) -> None:
        cursor = block.get_cursor()

        chapter = Chapter.get_latest_started_chapter(cursor)
        if chapter is None:
            raise LLMSolvableError("No chapters have been started yet")

        logger.debug(f"Reading wiki page '{self.slug}' at chapter {chapter.id}")
        page = WikiPage.read_page_at(cursor, self.slug, chapter.id)

        if page is None:
            # Find similar pages to suggest alternatives
            similar_pages = find_similar_slugs(cursor, self.slug, chapter.id)

            if similar_pages:
                suggestions = []
                for page in similar_pages:
                    suggestions.append(f"  - {page.title} ({page.slug}) {page.summary}")

                error_msg = (
                    f"No page exists with slug '{self.slug}'.\n\n"
                    "Did you mean one of these?\n"
                )
                error_msg += "\n".join(suggestions)
            else:
                error_msg = f"No page exists with slug '{self.slug}'"

            raise LLMSolvableError(error_msg)

        text = (
            f"# {page.title}\nKnown names: {page.names}\n"
            + f"Summary: {page.summary}\n\n{page.body}"
        )
        block.respond(text)


class WriteWikiPage(ToolModel):
    """Create a new wiki page or update an existing one with comprehensive information
    about characters, locations, or concepts.

    For NEW pages: Set create=true and provide ALL fields (title, names, summary,
    body). For UPDATES: Set create=false and provide only the fields you want to
    change (others will be preserved).

    Write thorough, spoiler-free content that helps readers track information up to
    the current chapter. Include cross-references to related pages using Markdown
    links [display text](slug).

    IMPORTANT: This tool validates all wiki links before writing. If any broken links
    are detected (links to non-existent pages), the operation will fail with an error
    listing the broken links.
    """

    slug: str
    """Unique identifier for the wiki page (e.g., 'aragorn-son-of-arathorn',
    'rivendell'). Use lowercase with hyphens. Keep consistent - this is permanent."""

    title: Optional[str]
    """Display name for the page (e.g., 'Aragorn, Son of Arathorn', 'Rivendell').
    Required when creating new pages."""

    names: Optional[list[str]]
    """All names, aliases, titles, and nicknames mentioned in text (e.g., ['Aragorn',
    'Strider', 'King Elessar']). Required when creating new pages. Include every
    variation for searchability.

    IMPORTANT: Names will be normalized, removing leading 'the ', and selecting only
    one capitalization of the name; as well as other possible modifications. This
    is expected behavior to improve the quality of search using the names. Provide
    names as best possible, but do not be surprised when the names on a page differ
    from the names you provide."""

    summary: Optional[str]
    """One or two sentence overview for quick identification (e.g., 'A mysterious
    Ranger who guides the hobbits from Bree'). Required when creating new pages.
    Use plain text only - no markdown formatting."""

    body: Optional[str]
    """Main content in Markdown format. Include: first appearance, key relationships,
    important actions, descriptions, current status, and cross-references. Required
    when creating new pages."""

    create: bool
    """Set to true when creating a completely new wiki page. Set to false when
    updating an existing page. Creating requires all fields; updating only changes
    provided fields."""

    delete_and_redirect_to: Optional[str] = None
    """Set to a slug to delete this page and redirect all links pointing to it.
    When set, this will: 1) validate the target slug exists, 2) find all wiki pages
    with links to the deleted page, 3) update those links to point to the redirect
    target, 4) mark the page as deleted. This is useful for consolidating duplicate
    pages or renaming pages while maintaining link integrity."""

    def _apply(self, block: Block) -> None:
        cursor = block.get_cursor()

        # Get the latest started chapter - this is where we write
        latest_chapter = Chapter.get_latest_started_chapter(cursor)
        if latest_chapter is None:
            raise LLMSolvableError("No chapters have been started yet")
        chapter_id = latest_chapter.id

        existing_page = WikiPage.read_page_at(cursor, self.slug, chapter_id)

        # Handle deletion with redirect request
        if self.delete_and_redirect_to is not None:
            if existing_page is None:
                raise LLMSolvableError(
                    "Cannot delete non-existent page - no such slug exists"
                )

            # Validate redirect target exists and is different from current slug
            if self.delete_and_redirect_to == self.slug:
                raise LLMSolvableError(
                    "Cannot redirect a page to itself. "
                    "The redirect target must be a different page."
                )

            # Allow empty string to unlink references without redirecting
            if self.delete_and_redirect_to != "":
                target_page = WikiPage.read_page_at(
                    cursor, self.delete_and_redirect_to, chapter_id
                )
                if target_page is None:
                    # Find similar pages to suggest alternatives
                    similar_pages = find_similar_slugs(
                        cursor, self.delete_and_redirect_to, chapter_id
                    )

                    if similar_pages:
                        suggestions = []
                        for page in similar_pages:
                            suggestions.append(f"  - {page.title} ({page.slug})")

                        error_msg = (
                            f"Cannot redirect to non-existent page "
                            f"'{self.delete_and_redirect_to}'.\n\n"
                            "Did you mean one of these?\n"
                        )
                        error_msg += "\n".join(suggestions)
                    else:
                        error_msg = (
                            f"Cannot redirect to non-existent page "
                            f"'{self.delete_and_redirect_to}'"
                        )

                    raise LLMSolvableError(error_msg)

            # Validate that other content fields are not set when deleting
            has_title = self.title is not None and self.title.strip() != ""
            has_names = (
                self.names is not None
                and len(self.names) > 0
                and any(name.strip() != "" for name in self.names)
            )
            has_summary = self.summary is not None and self.summary.strip() != ""
            has_body = self.body is not None and self.body.strip() != ""

            if has_title or has_names or has_summary or has_body:
                raise LLMSolvableError(
                    "Cannot set content fields (title, names, summary, body) when "
                    "using delete_and_redirect_to. Either set delete_and_redirect_to "
                    "or provide content, not both."
                )

            logger.info(
                f"Deleting wiki page '{self.slug}' and redirecting to "
                f"'{self.delete_and_redirect_to}' "
                f"at chapter {chapter_id}"
            )

            # Use the WikiPage method for deletion and redirection
            pages_updated, response = existing_page.delete_and_redirect(
                block, self.delete_and_redirect_to
            )

            block.respond(response)
            return

        if self.create:
            if existing_page is not None:
                raise LLMSolvableError(
                    "That slug already exists, but create was specified"
                )
            logger.info(f"Creating new wiki page '{self.slug}' at chapter {chapter_id}")

            if not (self.title and self.names and self.summary and self.body):
                raise LLMSolvableError(
                    "All fields must be set when creating a new page"
                )

            # Check for linting issues before writing
            lint_warnings = self._check_links_before_write(
                cursor, self.body, chapter_id
            )
            if lint_warnings:
                error_msg = "Cannot create wiki page with broken links:\n" + "\n".join(
                    lint_warnings
                )
                raise LLMSolvableError(error_msg)

            block.write_wiki_page(
                chapter_id,
                self.slug,
                self.title,
                self.names,
                self.summary,
                self.body,
            )
        else:
            if existing_page is None:
                raise LLMSolvableError(
                    "No such slug exists, but create was not specified"
                )
            logger.info(f"Updating wiki page '{self.slug}' at chapter {chapter_id}")

            # Check for linting issues before writing
            body_to_check = self.body if self.body is not None else existing_page.body
            lint_warnings = self._check_links_before_write(
                cursor, body_to_check, chapter_id
            )
            if lint_warnings:
                error_msg = "Cannot update wiki page with broken links:\n" + "\n".join(
                    lint_warnings
                )
                raise LLMSolvableError(error_msg)

            block.write_wiki_page(
                chapter_id,
                self.slug,
                self.title or existing_page.title,
                self.names or existing_page.names,
                self.summary or existing_page.summary,
                self.body or existing_page.body,
            )

        block.respond("Wrote wiki page")

    def _check_links_before_write(
        self, cursor: Cursor, body: str, chapter_id: int
    ) -> list[str]:
        """Check for broken wiki links in body content before writing.

        Args:
            cursor: Database cursor
            body: The body content to check
            chapter_id: The chapter ID

        Returns:
            List of warning messages for broken links
        """
        warnings: list[str] = []

        # Get all valid slugs for efficiency
        valid_slugs = WikiPage.get_all_slugs(cursor, chapter_id)

        # Extract wiki links from the body
        wiki_links = extract_wiki_links(body)

        # Check each link to see if it references a valid wiki page
        for link in wiki_links:
            if link.slug not in valid_slugs:
                warnings.append(
                    f"- Slug '{link.slug}' does not reference a page in the wiki"
                )

        return warnings


class SearchWikiByName(ToolModel):
    """Find existing wiki pages by searching for character names, location names, or
    any other identifiers.

    Use this tool to discover which wiki pages already exist for entities mentioned
    in the text. The search uses fuzzy matching to find pages even with slight
    spelling variations or partial names. Results are ranked by relevance and include
    the page slug, known names, and summary.

    IMPORTANT: Each search is for ONE entity only. Provide multiple names/aliases
    for the same character, location, or concept. To find multiple different
    entities, use multiple separate SearchWikiByName calls.
    """

    results_page: Optional[int]
    """Which page of results to return (default: 1). Each page shows up to 6
    results. Use higher numbers to see more results if many matches are found."""

    names: list[str]
    """List of names/aliases that refer to a SINGLE entity you want to find (e.g.,
    ['Gandalf', 'Grey Wizard', 'Mithrandir'] to search for one character). All names
    provided should be different ways to refer to the same person, place, or concept.

    To search for multiple different entities, make separate SearchWikiByName calls
    - one per entity you want to find."""

    def _apply(self, block: Block) -> None:
        cursor = block.get_cursor()

        chapter = Chapter.get_latest_started_chapter(cursor)
        if chapter is None:
            raise LLMSolvableError("No chapters have been started yet")

        page = self.results_page or 1
        assert page > 0

        search_results = search_wiki_by_name(
            cursor, page, self.names, chapter.id, page_size=6
        )

        results = []
        for result in search_results.results:
            # Format: rank/rrf_score/slug/names/summary
            names_str = ", ".join(result.page.names)
            result_line = (
                f"{result.rank}. {result.page.title} - {result.page.slug}\n"
                f"   Names: {names_str}\n"
                f"   Summary: {result.page.summary}"
            )
            results.append(result_line)

        if not results:
            if page == 1:
                block.respond("No wiki pages found.")
            else:
                block.respond(f"No results found on page {page}")
            return

        # Format final response
        response = (
            f"Search Results (Page {page}, showing {len(results)} "
            f"of {search_results.total_results} total):\n\n"
        )
        response += "\n\n".join(results)

        block.respond(response)
