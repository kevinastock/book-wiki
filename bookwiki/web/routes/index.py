"""Index/home page route."""

from flask import Blueprint, render_template

from bookwiki.models.block import Block, ToolUsageStats
from bookwiki.models.conversation import Conversation
from bookwiki.tools import get_all_tools
from bookwiki.web.app import get_db

index_bp = Blueprint("index", __name__)


@index_bp.route("/")
def index() -> str:
    """Root welcome page with stats dashboard."""
    with get_db().transaction_cursor() as cursor:
        # Get all statistics including conversation status counts
        conversations_stats = Conversation.get_all_conversations_stats(cursor)

        # Get count of unresponded feedback requests using Block method
        feedback_count = len(
            Block.get_unresponded_blocks(cursor, "RequestExpertFeedback")
        )

        # Get tool usage statistics
        tool_stats = Block.get_tool_usage_stats(cursor)

        # Get conversation length percentiles
        conv_length_percentiles = (
            Conversation.get_finished_conversations_token_percentiles(cursor)
        )

        # Get all available tools and ensure they all appear in stats (with 0s)
        all_tools = get_all_tools()
        all_tool_names = {tool_class.__name__ for tool_class in all_tools}
        tool_stats_names = {stat.name for stat in tool_stats}

        # Add missing tools with zero stats
        for tool_name in all_tool_names - tool_stats_names:
            tool_stats.append(ToolUsageStats(name=tool_name, used=0, failed=0))

        # Sort by tool name for consistent display
        tool_data = sorted(tool_stats, key=lambda x: x.name)

        stats = {
            "total_input_tokens": conversations_stats.total_input_tokens,
            "total_output_tokens": conversations_stats.total_output_tokens,
            "feedback_requests": feedback_count,
            "conversations_waiting_llm": (
                conversations_stats.conversations_waiting_llm
            ),
            "conversations_waiting_tools": (
                conversations_stats.conversations_waiting_tools
            ),
            "conversations_ready": conversations_stats.conversations_ready,
            "conversations_finished": conversations_stats.conversations_finished,
        }

    return render_template(
        "index.html",
        stats=stats,
        tool_data=tool_data,
        conv_lengths=conv_length_percentiles,
    )
