"""Chapter routes for the bookwiki web interface."""

from flask import Blueprint, abort, render_template

from bookwiki.models import Chapter
from bookwiki.web.app import get_db
from bookwiki.web.filters import format_chapter_title

chapters_bp = Blueprint("chapters", __name__)


@chapters_bp.route("/chapters")
def list_chapters() -> str:
    """List all chapters with basic info and wiki page counts."""
    with get_db().transaction_cursor() as cursor:
        # Get all chapters with wiki page counts
        # TODO: paginate properly, maybe by book?
        chapters_with_counts = Chapter.get_page_of_chapters(cursor, 0, 10000)

        chapters = []
        for chapter, wiki_count in chapters_with_counts:
            title = format_chapter_title(chapter)
            chapters.append(
                {"id": chapter.id, "title": title, "wiki_count": wiki_count}
            )

    return render_template("chapters/list.html", chapters=chapters)


@chapters_bp.route("/chapter/<int:chapter_id>")
def chapter_detail(chapter_id: int) -> str:
    """Show chapter text and associated wiki pages."""
    # Get wiki pages created/updated in this chapter
    with get_db().transaction_cursor() as cursor:
        # Get the chapter
        chapter = Chapter.read_chapter(cursor, chapter_id)
        if not chapter:
            abort(404)

        # Split chapter text into paragraphs for better display
        # Each line is a paragraph in the source text
        if chapter.text:
            paragraphs = [p.strip() for p in chapter.text.split("\n") if p.strip()]
        else:
            paragraphs = []

        return render_template(
            "chapters/detail.html",
            chapter=chapter,
            paragraphs=paragraphs,
        )
