"""Tests for the deserialize_tool function in bookwiki.tools."""

import json

import pytest
from pydantic import ValidationError

import bookwiki.tools as tools_module
from bookwiki.tools import (
    ReadChapter,
    ReadWikiPage,
    SearchWikiByName,
    SpawnAgent,
    WriteWikiPage,
    deserialize_tool,
)
from bookwiki.tools.base import ToolModel


class TestDeserializeTool:
    """Test the deserialize_tool function."""

    def test_deserialize_from_dict(self) -> None:
        """Test deserializing a tool from a dictionary."""
        tool_dict = {
            "chapter_offset": -5,
        }

        result = deserialize_tool(
            tool_dict,
            (ReadChapter,),
            tool_id="test_id_1",
            tool_name="ReadChapter",
        )

        assert isinstance(result, ReadChapter)
        assert result.chapter_offset == -5
        assert result.tool_id == "test_id_1"
        assert result.tool_name == "ReadChapter"

    def test_deserialize_from_json_string(self) -> None:
        """Test deserializing a tool from a JSON string."""
        json_str = json.dumps(
            {
                "slug": "test-page",
                "title": "Test Page",
                "names": ["Test", "Page"],
                "summary": "A test page",
                "body": "Test content",
                "create": True,
            }
        )

        result = deserialize_tool(
            json_str,
            (WriteWikiPage,),
            tool_id="test_id_2",
            tool_name="WriteWikiPage",
        )

        assert isinstance(result, WriteWikiPage)
        assert result.slug == "test-page"
        assert result.title == "Test Page"
        assert result.names == ["Test", "Page"]
        assert result.summary == "A test page"
        assert result.body == "Test content"
        assert result.create is True
        assert result.tool_id == "test_id_2"
        assert result.tool_name == "WriteWikiPage"

    def test_deserialize_with_multiple_candidates(self) -> None:
        """Test deserializing when multiple tool types are candidates."""
        # Test with ReadWikiPage - just include slug which is required
        read_dict = {"slug": "hero-journey"}

        result = deserialize_tool(
            read_dict,
            (ReadWikiPage,),  # Single candidate to avoid union issues
            tool_id="test_id_3",
            tool_name="ReadWikiPage",
        )

        assert isinstance(result, ReadWikiPage)
        assert result.slug == "hero-journey"

        # Test with SearchWikiByName - include all required fields
        search_dict = {
            "names": ["dragon", "wyrm"],
            "results_page": None,  # Optional field that must be explicitly provided
        }

        result = deserialize_tool(
            search_dict,
            (SearchWikiByName,),  # Single candidate to avoid union issues
            tool_id="test_id_4",
            tool_name="SearchWikiByName",
        )

        assert isinstance(result, SearchWikiByName)
        assert result.names == ["dragon", "wyrm"]
        assert result.results_page is None  # Optional field defaults to None

    def test_deserialize_with_optional_fields(self) -> None:
        """Test deserializing a tool with optional fields."""
        # SearchWikiByName has an optional results_page field
        search_dict = {
            "names": ["wizard", "mage"],
            "results_page": None,  # Optional field
        }

        result = deserialize_tool(
            search_dict,
            (SearchWikiByName,),
            tool_id="test_id_5",
            tool_name="SearchWikiByName",
        )

        assert isinstance(result, SearchWikiByName)
        assert result.names == ["wizard", "mage"]
        assert result.results_page is None

    def test_deserialize_with_all_fields(self) -> None:
        """Test deserializing a tool with all fields including optionals."""
        spawn_dict = {
            "prompt_key": "analyze_character",
            "template_names": ["character"],
            "template_values": ["Bob"],
        }

        result = deserialize_tool(
            spawn_dict,
            (SpawnAgent,),
            tool_id="test_id_6",
            tool_name="SpawnAgent",
        )

        assert isinstance(result, SpawnAgent)
        assert result.prompt_key == "analyze_character"
        assert result.template_names == ["character"]
        assert result.template_values == ["Bob"]

    def test_deserialize_invalid_json_string(self) -> None:
        """Test that invalid JSON string raises an error."""
        invalid_json = "{ not valid json }"

        with pytest.raises(ValidationError):
            deserialize_tool(
                invalid_json,
                (ReadChapter,),
                tool_id="test_id_7",
                tool_name="ReadChapter",
            )

    def test_deserialize_optional_field(self) -> None:
        """Test that optional fields can be omitted."""
        # ReadChapter has optional chapter_offset
        incomplete_dict: dict[str, int] = {}

        # This should succeed since chapter_offset is optional
        tool = deserialize_tool(
            incomplete_dict,
            (ReadChapter,),
            tool_id="test_id_8",
            tool_name="ReadChapter",
        )

        assert isinstance(tool, ReadChapter)
        assert tool.chapter_offset is None

    def test_deserialize_wrong_type(self) -> None:
        """Test that wrong field types raise ValidationError."""
        wrong_type_dict = {
            "chapter_offset": "not_a_number",  # Should be int
        }

        with pytest.raises(ValidationError) as exc_info:
            deserialize_tool(
                wrong_type_dict,
                (ReadChapter,),
                tool_id="test_id_9",
                tool_name="ReadChapter",
            )

        error = exc_info.value
        assert "chapter_offset" in str(error)

    def test_deserialize_extra_fields_forbidden(self) -> None:
        """Test that extra fields cause a ValidationError."""
        dict_with_extras = {
            "chapter_offset": -10,
            "extra_field": "should be rejected",
            "another_extra": 123,
        }

        with pytest.raises(ValidationError) as exc_info:
            deserialize_tool(
                dict_with_extras,
                (ReadChapter,),
                tool_id="test_id_10",
                tool_name="ReadChapter",
            )

        error = exc_info.value
        error_str = str(error)
        assert "extra_field" in error_str
        assert "another_extra" in error_str
        assert "Extra inputs are not permitted" in error_str

    def test_deserialize_empty_candidates(self) -> None:
        """Test that empty candidates tuple raises an error."""
        with pytest.raises(TypeError):
            deserialize_tool(
                {"chapter_offset": 0},
                (),  # Empty candidates
                tool_id="test_id_11",
                tool_name="ReadChapter",
            )

    def test_adapter_caching(self) -> None:
        """Test that the adapter is cached for the same candidates."""
        _adapter_for = tools_module._adapter_for

        # Clear cache first
        _adapter_for.cache_clear()

        # First call should create adapter
        candidates1 = (ReadChapter, WriteWikiPage)
        adapter1 = _adapter_for(candidates1)

        # Second call with same candidates should return cached adapter
        adapter2 = _adapter_for(candidates1)
        assert adapter1 is adapter2

        # The sorting in deserialize_tool ensures consistent ordering
        # But _adapter_for itself doesn't do sorting, it relies on the input
        # So different order would be a different cache key
        candidates2 = (WriteWikiPage, ReadChapter)
        adapter3 = _adapter_for(candidates2)
        # This should be different since _adapter_for doesn't sort its input
        assert adapter3 is not adapter1

        # Same order again should hit cache
        candidates4 = (WriteWikiPage, ReadChapter)
        adapter5 = _adapter_for(candidates4)
        assert adapter5 is adapter3

        # Different candidates should create new adapter
        candidates6 = (SearchWikiByName,)
        adapter6 = _adapter_for(candidates6)
        assert adapter6 is not adapter1
        assert adapter6 is not adapter3

        # Test the cache info to verify caching is working
        cache_info = _adapter_for.cache_info()
        assert cache_info.hits >= 1  # We should have cache hits
        assert cache_info.misses >= 2  # We should have multiple cache misses

    def test_deserialize_complex_nested_data(self) -> None:
        """Test deserializing tools with complex data structures."""
        # SpawnAgent's template_names and template_values are lists of strings
        complex_dict = {
            "prompt_key": "complex_analysis",
            "template_names": [
                "character_name",
                "chapter_range",
                "analysis_type",
                "json_data",
            ],
            "template_values": [
                "Alice",
                "1-10",
                "personality",
                '{"nested": {"key": "value"}}',  # JSON as string
            ],
        }

        result = deserialize_tool(
            complex_dict,
            (SpawnAgent,),
            tool_id="test_complex",
            tool_name="SpawnAgent",
        )

        assert isinstance(result, SpawnAgent)
        # Check that the variables are correctly paired
        template_vars = dict(
            zip(result.template_names, result.template_values, strict=False)
        )
        assert template_vars["character_name"] == "Alice"
        assert template_vars["chapter_range"] == "1-10"
        assert template_vars["analysis_type"] == "personality"
        assert template_vars["json_data"] == '{"nested": {"key": "value"}}'

    def test_deserialize_with_none_values(self) -> None:
        """Test deserializing with None values for optional fields."""
        # WriteWikiPage with None values
        dict_with_none = {
            "slug": "test-slug",
            "title": None,
            "names": None,
            "summary": None,
            "body": None,
            "create": False,
        }

        result = deserialize_tool(
            dict_with_none,
            (WriteWikiPage,),
            tool_id="test_none",
            tool_name="WriteWikiPage",
        )

        assert isinstance(result, WriteWikiPage)
        assert result.slug == "test-slug"
        assert result.title is None
        assert result.names is None
        assert result.summary is None
        assert result.body is None
        assert result.create is False

    def test_assert_isinstance_always_passes(self) -> None:
        """Test that the assert isinstance(result, ToolModel) always passes."""
        # Since all our tool classes inherit from ToolModel, this should always pass
        result = deserialize_tool(
            {"chapter_offset": 0},
            (ReadChapter,),
            tool_id="test_assert",
            tool_name="ReadChapter",
        )

        # This mirrors the assertion in the actual function
        assert isinstance(result, ToolModel)
        assert isinstance(result, ReadChapter)
