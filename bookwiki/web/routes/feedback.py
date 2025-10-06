"""Feedback requests route."""

from flask import Blueprint, flash, redirect, render_template, request, url_for
from werkzeug.wrappers import Response

from bookwiki.models import Block
from bookwiki.web.app import get_db

feedback_bp = Blueprint("feedback", __name__)


@feedback_bp.route("/feedback")
def feedback_list() -> str:
    """List all feedback requests from RequestExpertFeedback tool uses."""
    with get_db().transaction_cursor() as cursor:
        return render_template(
            "feedback.html",
            feedback_blocks=Block.get_unresponded_blocks(
                cursor, "RequestExpertFeedback"
            ),
        )


@feedback_bp.route("/feedback/<int:block_id>/submit", methods=["POST"])
def submit_feedback(block_id: int) -> Response:
    """Submit a feedback response to a RequestExpertFeedback block."""
    response_text = request.form.get("response", "").strip()

    if not response_text:
        flash("Response cannot be empty", "error")
        return redirect(url_for("feedback.feedback_list"))

    try:
        with get_db().transaction_cursor() as cursor:
            # Get the block using Block.get_by_id
            block = Block.get_by_id(cursor, block_id)

            if block is None:
                flash("Feedback request not found", "error")
                return redirect(url_for("feedback.feedback_list"))

            # Verify it's a RequestExpertFeedback block
            if block.tool_name != "RequestExpertFeedback":
                flash("Invalid feedback request", "error")
                return redirect(url_for("feedback.feedback_list"))

            # Check if already responded
            if block.tool_response:
                flash("This feedback request has already been responded to", "warning")
                return redirect(url_for("feedback.feedback_list"))

            # Submit the response
            block.respond(response_text)

            flash("Feedback submitted successfully", "success")
            return redirect(url_for("feedback.feedback_list"))

    except Exception as e:
        flash(f"Failed to submit feedback: {str(e)}", "error")
        return redirect(url_for("feedback.feedback_list"))
