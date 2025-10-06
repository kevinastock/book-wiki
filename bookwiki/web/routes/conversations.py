"""Conversation routes for the web interface."""

from flask import Blueprint, abort, render_template

from bookwiki.models import Conversation
from bookwiki.web.app import get_db

conversations_bp = Blueprint("conversations", __name__)


@conversations_bp.route("/conversations")
def list_conversations() -> str:
    """List all root conversations."""
    with get_db().transaction_cursor() as cursor:
        root_conversations = Conversation.get_root_conversations(cursor)
        return render_template(
            "conversations/list.html", conversations=root_conversations
        )


@conversations_bp.route("/conversation/<int:conversation_id>")
def conversation_detail(conversation_id: int) -> str:
    """Show detailed view of a conversation with all blocks."""
    with get_db().transaction_cursor() as cursor:
        conversation = Conversation.get_by_id(cursor, conversation_id)
        if not conversation:
            abort(404)

        return render_template(
            "conversations/detail.html",
            conversation=conversation,
        )
