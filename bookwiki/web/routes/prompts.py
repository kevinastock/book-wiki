"""Prompt listing and detail routes."""

from flask import Blueprint, abort, render_template

from bookwiki.models import Prompt
from bookwiki.web.app import get_db

prompts_bp = Blueprint("prompts", __name__)


@prompts_bp.route("/prompts")
def prompts_list() -> str:
    """List all prompts (latest versions only)."""
    with get_db().transaction_cursor() as cursor:
        prompts = sorted(Prompt.list_prompts(cursor).values(), key=lambda p: p.key)
        return render_template("prompts/list.html", prompts=prompts)


@prompts_bp.route("/prompt/<key>")
def prompt_detail(key: str) -> str:
    """Show latest version of a prompt."""
    with get_db().transaction_cursor() as cursor:
        versions = Prompt.get_all_versions(cursor, key)

        if not versions:
            abort(404)

        return render_template("prompts/detail.html", versions=versions)
