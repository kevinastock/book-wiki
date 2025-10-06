"""Tool usage routes."""

from flask import Blueprint, abort, render_template, request

from bookwiki.models.block import Block, ToolUsageStats
from bookwiki.tools import get_all_tools
from bookwiki.web.app import get_db

tools_bp = Blueprint("tools", __name__)


@tools_bp.route("/tools/<tool_name>")
def tool_usage(tool_name: str) -> str:
    """Display usage details for a specific tool."""
    # Validate that this is a real tool
    all_tools = get_all_tools()
    tool_names = [tool.__name__ for tool in all_tools]

    if tool_name not in tool_names:
        abort(404)

    # Get page from query params
    page = int(request.args.get("page", 1))
    if page < 1:
        page = 1

    with get_db().transaction_cursor() as cursor:
        # Get paginated blocks for this tool
        paginated_result = Block.get_blocks_by_tool_paginated(
            cursor, tool_name, page, page_size=20
        )

        # Get usage statistics for this specific tool
        tool_stats = Block.get_tool_usage_stats(cursor)
        # Find the stats for this specific tool
        for stat in tool_stats:
            if stat.name == tool_name:
                stats = stat
                break
        else:
            # If not found, create a default one
            stats = ToolUsageStats(name=tool_name, used=0, failed=0)

    return render_template(
        "tools/usage.html",
        tool_name=tool_name,
        blocks=paginated_result.blocks,
        stats=stats,
        total_count=paginated_result.total_count,
        current_page=page,
        total_pages=paginated_result.total_pages,
    )
