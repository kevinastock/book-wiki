from flask import (
    Blueprint,
    abort,
    redirect,
    render_template,
    request,
    url_for,
)
from werkzeug.wrappers import Response

from bookwiki.models.chapter import Chapter
from bookwiki.models.wikipage import WikiPage
from bookwiki.search import search_wiki_by_name
from bookwiki.web.app import get_db

wiki_bp = Blueprint("wiki", __name__)


@wiki_bp.route("/wiki")
def wiki_redirect() -> Response:
    """Redirect to the latest started chapter's wiki page."""
    with get_db().transaction_cursor() as cursor:
        latest_chapter = Chapter.get_latest_started_chapter(cursor)
        if not latest_chapter:
            abort(404)

    return redirect(url_for("wiki.wiki_list", chapter_id=latest_chapter.id))


@wiki_bp.route("/wiki/<int:chapter_id>")
def wiki_list(chapter_id: int) -> str:
    """List all wiki pages visible at the given chapter."""
    with get_db().transaction_cursor() as cursor:
        # Verify chapter exists and is started
        chapter = Chapter.read_chapter(cursor, chapter_id)
        if not chapter or chapter.conversation_id is None:
            abort(404)

        pages = WikiPage.get_all_pages_chapter(cursor, chapter_id)
        available_chapters = Chapter.get_started_chapter_names(cursor)

        return render_template(
            "wiki/list.html",
            pages=pages,
            available_chapters=available_chapters,
            current_chapter=chapter_id,
            chapter_nav_url_template="/wiki/{chapter_id}",
        )


@wiki_bp.route("/wiki/<int:chapter_id>/<path:slug>")
def wiki_page(chapter_id: int, slug: str) -> str:
    """Display a wiki page for a given slug at a specific chapter."""
    with get_db().transaction_cursor() as cursor:
        page = WikiPage.read_page_at(cursor, slug, chapter_id)

        if not page:
            abort(404)

        # Get available chapters for nav
        available_chapters = Chapter.get_started_chapter_names(cursor)

        return render_template(
            "wiki/page.html",
            page=page,
            chapter_id=chapter_id,
            available_chapters=available_chapters,
            current_chapter=chapter_id,
            chapter_nav_url_template=f"/wiki/{{chapter_id}}/{slug}",
            is_historical=False,
        )


@wiki_bp.route("/search")
def search_redirect() -> Response:
    """Redirect to the latest started chapter's search page."""
    with get_db().transaction_cursor() as cursor:
        latest_chapter = Chapter.get_latest_started_chapter(cursor)
        if not latest_chapter:
            abort(404)

    # Preserve query parameters in the redirect
    base_url = url_for("wiki.wiki_search", chapter_id=latest_chapter.id)
    if request.query_string:
        base_url = f"{base_url}?{request.query_string.decode()}"
    return redirect(base_url)


@wiki_bp.route("/search/<int:chapter_id>")
def wiki_search(chapter_id: int) -> str:
    """Search wiki pages by name with fuzzy matching."""
    with get_db().transaction_cursor() as cursor:
        # Verify chapter exists and is started
        chapter = Chapter.read_chapter(cursor, chapter_id)
        if not chapter or chapter.conversation_id is None:
            abort(404)

        available_chapters = Chapter.get_started_chapter_names(cursor)

        names_param = request.args.get("names", "").strip()
        page = int(request.args.get("page", 1))

        if not names_param:
            return render_template(
                "wiki/search.html",
                results=None,
                search_names="",
                page=page,
                available_chapters=available_chapters,
                current_chapter=chapter_id,
                chapter_nav_url_template="/search/{chapter_id}",
            )

        # Split names by comma and clean up
        names = [name.strip() for name in names_param.split(",") if name.strip()]

        results = search_wiki_by_name(cursor, page, names, chapter_id)
        available_chapters = Chapter.get_started_chapter_names(cursor)

        # Build URL template preserving current search params
        query_params = []
        if names_param:
            query_params.append(f"names={names_param}")
        if page > 1:
            query_params.append(f"page={page}")
        query_string = "&".join(query_params)
        url_template = "/search/{chapter_id}" + (
            f"?{query_string}" if query_string else ""
        )

        return render_template(
            "wiki/search.html",
            results=results,
            search_names=names_param,
            page=page,
            names=names,
            available_chapters=available_chapters,
            current_chapter=chapter_id,
            chapter_nav_url_template=url_template,
        )


@wiki_bp.route("/history/<int:chapter_id>/<path:slug>")
def wiki_history_list(chapter_id: int, slug: str) -> str:
    """List all versions of a wiki page up to given chapter."""
    with get_db().transaction_cursor() as cursor:
        versions = WikiPage.get_versions_by_slug(cursor, slug, chapter_id)
        if not versions:
            abort(404)

        # Get available chapters for nav
        available_chapters = Chapter.get_started_chapter_names(cursor)

        return render_template(
            "wiki/history.html",
            slug=slug,
            versions=versions[::-1],
            current_chapter=chapter_id,
            available_chapters=available_chapters,
            chapter_nav_url_template=f"/history/{{chapter_id}}/{slug}",
        )


@wiki_bp.route("/pageid/<int:chapter_id>/<int:pageid>")
def wiki_page_by_id(chapter_id: int, pageid: int) -> str:
    """Display a specific historical version of a wiki page by ID."""
    with get_db().transaction_cursor() as cursor:
        wiki_page = WikiPage.get_by_id(cursor, pageid)
        if not wiki_page:
            abort(404)

        # Get available chapters for nav
        available_chapters = Chapter.get_started_chapter_names(cursor)

        return render_template(
            "wiki/page.html",
            page=wiki_page,
            chapter_id=chapter_id,
            available_chapters=available_chapters,
            current_chapter=chapter_id,
            chapter_nav_url_template=f"/pageid/{{chapter_id}}/{pageid}",
            is_historical=True,
        )
