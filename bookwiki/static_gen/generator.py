#!/usr/bin/env python3
"""Static site generator for FanWiki pages."""

import argparse
import asyncio
import logging
import shutil
import sqlite3
from pathlib import Path
from typing import Any

import markdown  # type: ignore
import minify_html
from jinja2 import Environment, FileSystemLoader
from pagefind import index

from bookwiki.models.chapter import Chapter, ChapterName
from bookwiki.models.wikipage import WikiPage
from bookwiki.utils import extract_wiki_links

logger = logging.getLogger(__name__)


def format_chapter_title(chapter: Chapter | ChapterName) -> str:
    """Format a chapter title from its name list."""
    if chapter.name:
        return " - ".join(chapter.name)
    return f"Chapter {chapter.id}"


def markdown_summary_with_static_wiki_links(
    text: str | None, chapter_id: int, base_url: str = ""
) -> str | None:
    """Convert Markdown summary to HTML with wiki links transformed to static URLs.

    This processes markdown the same as markdown_with_static_wiki_links but adds
    Bootstrap utility classes to remove bottom margins for better card layouts.

    Args:
        text: The Markdown text to convert
        chapter_id: Current chapter for generating wiki URLs
        base_url: Base URL to prepend to wiki links

    Returns:
        HTML with wiki links transformed to {base_url}/wiki/{chapter_id}/{slug} format
        and Bootstrap classes added to remove margins
    """
    html = markdown_with_static_wiki_links(text, chapter_id, base_url)
    if html:
        # Add mb-0 class to paragraph tags to remove bottom margin in Bootstrap layouts
        html = html.replace("<p>", '<p class="mb-0">')
    return html


def markdown_with_static_wiki_links(
    text: str | None, chapter_id: int, base_url: str = ""
) -> str | None:
    """Convert Markdown to HTML with wiki links transformed to static URLs.

    Args:
        text: The Markdown text to convert
        chapter_id: Current chapter for generating wiki URLs
        base_url: Base URL to prepend to wiki links

    Returns:
        HTML with wiki links transformed to {base_url}/wiki/{chapter_id}/{slug} format
    """
    if not text:
        return ""

    # Strip trailing slash from base_url to avoid double slashes
    base_url = base_url.rstrip("/")

    # Extract all wiki links using the shared utility
    wiki_links = extract_wiki_links(text)

    # Transform text by replacing each original link with chapter-aware version
    transformed_text = text
    for link in wiki_links:
        original_link = f"[{link.display_text}]({link.target})"
        new_link = f"[{link.display_text}]({base_url}/wiki/{chapter_id}/{link.slug})"
        transformed_text = transformed_text.replace(original_link, new_link)

    # Convert to HTML using markdown
    html_content: str = markdown.markdown(transformed_text)
    return html_content


def get_latest_version_per_chapter(
    cursor: sqlite3.Cursor, slug: str, up_to_chapter: int
) -> list[dict[str, Any]]:
    """Get the latest version of a wiki page for each chapter where it was updated.

    Args:
        cursor: Database cursor
        slug: The slug of the wiki page
        up_to_chapter: Maximum chapter to consider

    Returns:
        List of dicts with chapter info and the latest version for that chapter
    """
    # Get all versions grouped by chapter, taking only the latest per chapter
    rows = cursor.execute(
        """
        WITH latest_per_chapter AS (
            SELECT
                wp.*,
                ROW_NUMBER() OVER (
                    PARTITION BY wp.chapter ORDER BY wp.create_time DESC
                ) as rn
            FROM wiki_page wp
            WHERE wp.slug = ? AND wp.chapter <= ?
        )
        SELECT * FROM latest_per_chapter
        WHERE rn = 1
        ORDER BY chapter DESC
        """,
        (slug, up_to_chapter),
    ).fetchall()

    versions = []
    for row in rows:
        chapter = Chapter.read_chapter(cursor, row["chapter"])
        if chapter:
            versions.append(
                {
                    "chapter_id": row["chapter"],
                    "chapter_title": format_chapter_title(chapter),
                }
            )

    return versions


def minify_html_content(html: str) -> str:
    """Minify HTML content while preserving structure.

    Args:
        html: The HTML content to minify

    Returns:
        Minified HTML content
    """
    try:
        return minify_html.minify(
            html,
            minify_js=True,
            minify_css=True,
            keep_html_and_head_opening_tags=True,
            remove_processing_instructions=True,
        )
    except Exception as e:
        logger.warning(f"Failed to minify HTML: {e}")
        return html  # Return unminified HTML on error


class StaticSiteGenerator:
    """Generate static HTML files for wiki pages."""

    def __init__(
        self,
        db_path: str,
        output_dir: str,
        site_title: str,
        content_name: str,
        max_chapters: int | None = None,
        enable_playground: bool = False,
        base_url: str = "",
    ):
        """Initialize the generator.

        Args:
            db_path: Path to the SQLite database
            output_dir: Directory to write static files
            site_title: Title for the site navigation bar
            content_name: Name of the content covered in this wiki
            max_chapters: Maximum number of chapters to process (None for all)
            enable_playground: Whether to enable Pagefind playground files
            base_url: Base URL to prepend to all generated links (e.g., '/mysite')
        """
        self.db_path = db_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.site_title = site_title
        self.content_name = content_name
        self.max_chapters = max_chapters
        self.enable_playground = enable_playground
        self.base_url = base_url.rstrip("/")  # Remove trailing slash if present

        # Set up Jinja2 environment
        template_dir = Path(__file__).parent / "templates"
        self.env = Environment(loader=FileSystemLoader(template_dir))

    def get_db_connection(self) -> sqlite3.Connection:
        """Get a database connection with row factory."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def copy_static_files(self) -> None:
        """Copy all static files to output directory."""
        static_dir = Path(__file__).parent / "static"
        if static_dir.exists():
            shutil.copytree(static_dir, self.output_dir, dirs_exist_ok=True)
            logger.info("Copied static files to output directory")

    def generate_index_redirect(self) -> None:
        """Generate index.html that redirects to /wiki."""
        template = self.env.get_template("redirect.html")
        html = template.render(
            redirect_url=f"{self.base_url}/wiki", redirect_text="Wiki"
        )
        index_path = self.output_dir / "index.html"
        index_path.write_text(minify_html_content(html))
        logger.info("Generated index.html redirect")

    def generate_wiki_index(self, chapters: list[ChapterName]) -> None:
        """Generate /wiki index page with chapter list."""
        template = self.env.get_template("index.html")

        chapter_list = []
        for chapter in chapters:
            chapter_list.append(
                {
                    "id": chapter.id,
                    "title": format_chapter_title(chapter),
                    "url": f"{self.base_url}/wiki/{chapter.id}",
                }
            )

        html = template.render(
            chapters=chapter_list,
            current_chapter_title="All Chapters",  # Display text for index page
            site_title=self.site_title,
            content_name=self.content_name,
            base_url=self.base_url,
        )

        wiki_dir = self.output_dir / "wiki"
        wiki_dir.mkdir(exist_ok=True)
        index_path = wiki_dir / "index.html"
        index_path.write_text(minify_html_content(html))
        logger.info("Generated wiki index page")

    def generate_chapter_list(
        self,
        cursor: sqlite3.Cursor,
        chapter: ChapterName,
        all_chapters: list[ChapterName],
    ) -> None:
        """Generate wiki list page for a chapter."""
        template = self.env.get_template("list.html")

        # Get all wiki pages for this chapter
        pages_raw = WikiPage.get_all_pages_chapter(cursor, chapter.id)

        # Process each page to convert summary markdown to HTML
        pages = []
        for page in pages_raw:
            pages.append(
                {
                    "title": page.title,
                    "slug": page.slug,
                    "summary_html": markdown_summary_with_static_wiki_links(
                        page.summary, chapter.id, self.base_url
                    ),
                }
            )

        # Get the full chapter object to access chapter_summary_page
        full_chapter = Chapter.read_chapter(cursor, chapter.id)
        chapter_summary_page = None
        if full_chapter and full_chapter.chapter_summary_page:
            summary_page = full_chapter.chapter_summary_page
            chapter_summary_page = {
                "title": summary_page.title,
                "summary_html": markdown_summary_with_static_wiki_links(
                    summary_page.summary, chapter.id, self.base_url
                ),
                "body_html": markdown_with_static_wiki_links(
                    summary_page.body, chapter.id, self.base_url
                ),
                "names": summary_page.names,
            }

        # Prepare chapter navigation
        chapters_nav = []
        for ch in all_chapters:
            chapters_nav.append(
                {
                    "id": ch.id,
                    "title": format_chapter_title(ch),
                    "url": f"{self.base_url}/wiki/{ch.id}",
                }
            )

        html = template.render(
            pages=pages,
            chapter_id=chapter.id,
            current_chapter_title=format_chapter_title(chapter),
            chapters=chapters_nav,
            site_title=self.site_title,
            chapter_summary_page=chapter_summary_page,
            base_url=self.base_url,
        )

        chapter_dir = self.output_dir / "wiki" / str(chapter.id)
        chapter_dir.mkdir(parents=True, exist_ok=True)
        index_path = chapter_dir / "index.html"
        index_path.write_text(minify_html_content(html))
        logger.info(f"Generated chapter {chapter.id} list page")

    def generate_wiki_page(
        self,
        cursor: sqlite3.Cursor,
        page: WikiPage,
        chapter_id: int,
        all_chapters: list[ChapterName],
    ) -> None:
        """Generate a wiki page HTML file."""
        template = self.env.get_template("page.html")

        # Get chapter info
        chapter = Chapter.read_chapter(cursor, chapter_id)
        if not chapter:
            return

        # Get first chapter info
        first_chapter = Chapter.read_chapter(cursor, page.first_chapter.id)
        if not first_chapter:
            return

        # Prepare chapter navigation - link to page if exists, else chapter list
        chapters_nav = []
        for ch in all_chapters:
            # Check if page exists at this chapter
            page_at_chapter = WikiPage.read_page_at(cursor, page.slug, ch.id)
            if page_at_chapter and page_at_chapter.title != "":
                # Page exists - link to it
                url = f"{self.base_url}/wiki/{ch.id}/{page.slug}"
            else:
                # Page doesn't exist or is deleted - link to chapter list
                url = f"{self.base_url}/wiki/{ch.id}"

            chapters_nav.append(
                {
                    "id": ch.id,
                    "title": format_chapter_title(ch),
                    "url": url,
                }
            )

        # Convert markdown body to HTML with static wiki links
        body_html = markdown_with_static_wiki_links(
            page.body, chapter_id, self.base_url
        )

        # Convert markdown summary to HTML with static wiki links
        summary_html = markdown_summary_with_static_wiki_links(
            page.summary, chapter_id, self.base_url
        )

        html = template.render(
            page={
                "title": page.title,
                "slug": page.slug,
                "summary_html": summary_html,
                "names": page.names,
                "body_html": body_html,
                "chapter_id": page.chapter_id,
                "chapter_title": format_chapter_title(page.chapter),
                "first_chapter_id": first_chapter.id,
                "first_chapter_title": format_chapter_title(first_chapter),
            },
            chapter_id=chapter_id,
            current_chapter_title=format_chapter_title(chapter),
            chapters=chapters_nav,
            site_title=self.site_title,
            base_url=self.base_url,
        )

        page_dir = self.output_dir / "wiki" / str(chapter_id) / page.slug
        page_dir.mkdir(parents=True, exist_ok=True)
        page_path = page_dir / "index.html"
        page_path.write_text(minify_html_content(html))

    def generate_history_page(
        self,
        cursor: sqlite3.Cursor,
        slug: str,
        chapter_id: int,
        all_chapters: list[ChapterName],
    ) -> None:
        """Generate history page for a wiki page."""
        template = self.env.get_template("history.html")

        # Get the current page to get its title
        current_page = WikiPage.read_page_at(cursor, slug, chapter_id)
        if not current_page:
            return

        # Get latest version per chapter
        versions = get_latest_version_per_chapter(cursor, slug, chapter_id)

        # Get current chapter info
        chapter = Chapter.read_chapter(cursor, chapter_id)
        if not chapter:
            return

        # Prepare chapter navigation - link to history if exists, else chapter list
        chapters_nav = []
        for ch in all_chapters:
            # Check if page exists at this chapter
            page_at_chapter = WikiPage.read_page_at(cursor, slug, ch.id)
            if page_at_chapter and page_at_chapter.title != "":
                # Page exists - link to its history
                url = f"{self.base_url}/wiki/{ch.id}/{slug}/history"
            else:
                # Page doesn't exist or is deleted - link to chapter list
                url = f"{self.base_url}/wiki/{ch.id}"

            chapters_nav.append(
                {
                    "id": ch.id,
                    "title": format_chapter_title(ch),
                    "url": url,
                }
            )

        html = template.render(
            slug=slug,
            page_title=current_page.title,
            versions=versions,
            chapter_id=chapter_id,
            current_chapter_title=format_chapter_title(chapter),
            chapters=chapters_nav,
            site_title=self.site_title,
            base_url=self.base_url,
        )

        history_dir = self.output_dir / "wiki" / str(chapter_id) / slug / "history"
        history_dir.mkdir(parents=True, exist_ok=True)
        history_path = history_dir / "index.html"
        history_path.write_text(minify_html_content(html))

    async def generate_chapter_search_index(self, chapter_id: int) -> None:
        """Generate Pagefind search index for a specific chapter."""
        # Define the output path for this chapter's search index
        search_output_path = self.output_dir / f"pagefind-{chapter_id}"

        # Configure Pagefind to only index wiki pages for this chapter
        config = index.IndexConfig(
            root_selector="main",  # Only index content within <main> tags
            exclude_selectors=["nav", "footer"],  # Exclude navigation elements
            verbose=True,
            output_path=str(search_output_path),
            force_language="en",
            write_playground=self.enable_playground,
        )

        async with index.PagefindIndex(config=config) as pagefind_index:
            # Only index wiki page HTML files for this chapter
            chapter_wiki_dir = self.output_dir / "wiki" / str(chapter_id)

            # Find all wiki page directories
            # (but exclude history directories and the chapter index)
            for page_dir in chapter_wiki_dir.iterdir():
                if page_dir.is_dir() and page_dir.name != "history":
                    page_html = page_dir / "index.html"
                    if page_html.exists():
                        # Read the HTML content and add to search index
                        html_content = page_html.read_text(encoding="utf-8")
                        await pagefind_index.add_html_file(
                            content=html_content,
                            url=f"{self.base_url}/wiki/{chapter_id}/{page_dir.name}/",
                        )

            logger.info(f"Generated search index for chapter {chapter_id}")

    async def generate_all_search_indexes(self, chapters: list[ChapterName]) -> None:
        """Generate search indexes for all chapters."""
        tasks = []
        for chapter in chapters:
            tasks.append(self.generate_chapter_search_index(chapter.id))

        await asyncio.gather(*tasks)
        logger.info("Generated all chapter search indexes")

    def generate_root_404(self, chapters: list[ChapterName]) -> None:
        """Generate root 404.html page."""
        template = self.env.get_template("404.html")

        chapter_list = []
        for chapter in chapters:
            chapter_list.append(
                {
                    "id": chapter.id,
                    "title": format_chapter_title(chapter),
                    "url": f"{self.base_url}/wiki/{chapter.id}",
                }
            )

        html = template.render(
            chapters=chapter_list,
            site_title=self.site_title,
            content_name=self.content_name,
            base_url=self.base_url,
        )

        not_found_path = self.output_dir / "404.html"
        not_found_path.write_text(minify_html_content(html))
        logger.info("Generated root 404.html page")

    def generate_chapter_404(
        self,
        cursor: sqlite3.Cursor,
        chapter: ChapterName,
        chapters: list[ChapterName],
    ) -> None:
        """Generate 404.html page for a chapter directory."""
        template = self.env.get_template("list.html")

        chapter_list = []
        for ch in chapters:
            chapter_list.append(
                {
                    "id": ch.id,
                    "title": format_chapter_title(ch),
                    "url": f"{self.base_url}/wiki/{ch.id}",
                }
            )

        # Get all wiki pages for this chapter
        pages_raw = WikiPage.get_all_pages_chapter(cursor, chapter.id)

        # Process each page to convert summary markdown to HTML
        pages = []
        for page in pages_raw:
            pages.append(
                {
                    "title": page.title,
                    "slug": page.slug,
                    "summary_html": markdown_summary_with_static_wiki_links(
                        page.summary, chapter.id, self.base_url
                    ),
                }
            )

        # Get the full chapter object to access chapter_summary_page
        full_chapter = Chapter.read_chapter(cursor, chapter.id)
        chapter_summary_page = None
        if full_chapter and full_chapter.chapter_summary_page:
            summary_page = full_chapter.chapter_summary_page
            chapter_summary_page = {
                "title": summary_page.title,
                "summary_html": markdown_summary_with_static_wiki_links(
                    summary_page.summary, chapter.id, self.base_url
                ),
                "body_html": markdown_with_static_wiki_links(
                    summary_page.body, chapter.id, self.base_url
                ),
                "names": summary_page.names,
            }

        html = template.render(
            pages=pages,
            chapter_id=chapter.id,
            current_chapter_title=format_chapter_title(chapter),
            chapters=chapter_list,
            site_title=self.site_title,
            chapter_summary_page=chapter_summary_page,
            show_404_alert=True,
            base_url=self.base_url,
        )

        chapter_dir = self.output_dir / "wiki" / str(chapter.id)
        chapter_dir.mkdir(parents=True, exist_ok=True)
        not_found_path = chapter_dir / "404.html"
        not_found_path.write_text(minify_html_content(html))
        logger.info(f"Generated 404.html page for chapter {chapter.id}")

    def generate(self) -> None:
        """Generate the complete static site."""
        conn = self.get_db_connection()
        cursor = conn.cursor()

        try:
            # Get all started chapters
            chapters = Chapter.get_started_chapter_names(cursor)
            if not chapters:
                logger.warning("No started chapters found")
                return

            # Apply max_chapters limit if specified
            if self.max_chapters is not None:
                chapters = chapters[: self.max_chapters]
                logger.info(f"Limited to first {self.max_chapters} chapters")

            logger.info(f"Processing {len(chapters)} chapters")
            n_chapters = len(chapters)
            print(
                f"\n=== Starting static site generation for {n_chapters} chapters ===\n"
            )

            # Copy static files
            print("Copying static files...")
            self.copy_static_files()

            # Generate index redirect
            print("Generating index redirect...")
            self.generate_index_redirect()

            # Generate wiki index
            print("Generating wiki index...")
            self.generate_wiki_index(chapters)

            # Generate root 404 page
            print("Generating root 404 page...")
            self.generate_root_404(chapters)
            print()

            # Generate pages for each chapter
            for i, chapter in enumerate(chapters, 1):
                logger.info(f"Processing chapter {chapter.id}")
                print(f"Processing chapter {chapter.id} ({i}/{len(chapters)})...")

                # Generate chapter list page
                self.generate_chapter_list(cursor, chapter, chapters)

                # Generate chapter 404 page
                self.generate_chapter_404(cursor, chapter, chapters)

                # Get all pages for this chapter
                pages = WikiPage.get_all_pages_chapter(cursor, chapter.id)
                logger.info(f"  Found {len(pages)} pages for chapter {chapter.id}")

                # Generate each wiki page
                for page in pages:
                    self.generate_wiki_page(cursor, page, chapter.id, chapters)

                    # Generate history page
                    self.generate_history_page(cursor, page.slug, chapter.id, chapters)

                logger.info(f"  Generated {len(pages)} wiki pages and history pages")
                n_pages = len(pages)
                print(
                    f"  ✓ Chapter {chapter.id} complete: {n_pages} wiki pages generated"
                )

            # Generate search indexes for all chapters
            print(f"\nGenerating search indexes for all {len(chapters)} chapters...")
            logger.info("Generating search indexes...")
            asyncio.run(self.generate_all_search_indexes(chapters))

            logger.info("Static site generation complete!")
            print("\n=== Static site generation complete! ===\n")

        finally:
            conn.close()


def main() -> None:
    """Main entry point for the static site generator."""
    parser = argparse.ArgumentParser(description="Generate static wiki site")
    parser.add_argument("output_dir", help="Directory to output static files")
    parser.add_argument(
        "--db",
        default="bookwiki.db",
        help="Path to database file (default: bookwiki.db)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )
    parser.add_argument(
        "--title",
        default="Book Wiki",
        help="Title for the site navigation bar (default: Book Wiki)",
    )
    parser.add_argument(
        "--content-name",
        default="this content",
        help="Name of the content covered in this wiki (default: this content)",
    )
    parser.add_argument(
        "--max-chapters",
        type=int,
        help="Only process the first N chapters (default: all chapters)",
    )
    parser.add_argument(
        "--playground",
        action="store_true",
        help="Enable Pagefind playground files for interactive search testing",
    )
    parser.add_argument(
        "--baseurl",
        default="",
        help="Base URL to prepend to all generated links (e.g., '/mysite' or 'https://example.com/wiki')",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Generate the site
    generator = StaticSiteGenerator(
        args.db,
        args.output_dir,
        args.title,
        args.content_name,
        args.max_chapters,
        args.playground,
        args.baseurl,
    )
    generator.generate()


if __name__ == "__main__":
    main()
