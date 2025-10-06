from functools import lru_cache, reduce
from operator import or_
from typing import Any, Tuple, Type

from pydantic import TypeAdapter

from bookwiki.tools.base import ToolModel
from bookwiki.tools.chapter import ReadChapter
from bookwiki.tools.prompt import ListPrompts, ShowPrompt, WritePrompt
from bookwiki.tools.system import RequestExpertFeedback, SpawnAgent
from bookwiki.tools.wiki import ReadWikiPage, SearchWikiByName, WriteWikiPage


@lru_cache(maxsize=64)
def _adapter_for(candidates: Tuple[Type[ToolModel], ...]) -> TypeAdapter:
    return TypeAdapter(reduce(or_, candidates))


def deserialize_tool(
    json_like: str | dict[str, Any],
    candidates: Tuple[Type[ToolModel], ...],
    *,
    tool_id: str,
    tool_name: str,
) -> ToolModel:
    # Sort candidates to ensure consistent cache keys for _adapter_for
    sorted_candidates = tuple(sorted(candidates, key=lambda x: x.__name__))
    adapter = _adapter_for(sorted_candidates)

    context = {
        "tool_id": tool_id,
        "tool_name": tool_name,
    }

    if isinstance(json_like, str):
        result = adapter.validate_json(json_like, context=context)
    else:
        result = adapter.validate_python(json_like, context=context)

    assert isinstance(result, ToolModel)

    return result


def get_all_tools() -> tuple[type[ToolModel], ...]:
    """Get all tool model classes.

    Returns:
        Tuple of all tool model classes
    """
    return (
        ReadChapter,
        ListPrompts,
        ShowPrompt,
        WritePrompt,
        RequestExpertFeedback,
        SpawnAgent,
        ReadWikiPage,
        WriteWikiPage,
        SearchWikiByName,
    )


__all__ = [
    "ToolModel",
    "Tool",
    "get_all_tools",
    "deserialize_tool",
    # Individual tool classes
    "ReadChapter",
    "ListPrompts",
    "ShowPrompt",
    "WritePrompt",
    "RequestExpertFeedback",
    "SpawnAgent",
    "ReadWikiPage",
    "WriteWikiPage",
    "SearchWikiByName",
]
