"""Configuration management for the bookwiki application."""

from importlib.resources import files
from sqlite3 import Cursor

from bookwiki.config_enums import (
    OpenAIModel,
    OpenAIReasoningEffort,
    OpenAIServiceTier,
    OpenAIVerbosity,
)


class Configuration:
    """Configuration management using the database configuration table."""

    @staticmethod
    def get_openai_model(cursor: Cursor) -> OpenAIModel:
        """Get the OpenAI model configuration."""
        row = cursor.execute(
            "SELECT value FROM configuration WHERE key = ?", ("openai_model",)
        ).fetchone()
        value = row[0] if row else OpenAIModel.GPT_5.value
        return OpenAIModel(value)

    @staticmethod
    def set_openai_model(cursor: Cursor, value: OpenAIModel) -> None:
        """Set the OpenAI model configuration."""
        cursor.execute(
            "INSERT OR REPLACE INTO configuration (key, value) VALUES (?, ?)",
            ("openai_model", value.value),
        )

    @staticmethod
    def get_openai_verbosity(cursor: Cursor) -> OpenAIVerbosity:
        """Get the OpenAI verbosity configuration."""
        row = cursor.execute(
            "SELECT value FROM configuration WHERE key = ?", ("openai_verbosity",)
        ).fetchone()
        value = row[0] if row else OpenAIVerbosity.MEDIUM.value
        return OpenAIVerbosity(value)

    @staticmethod
    def set_openai_verbosity(cursor: Cursor, value: OpenAIVerbosity) -> None:
        """Set the OpenAI verbosity configuration."""
        cursor.execute(
            "INSERT OR REPLACE INTO configuration (key, value) VALUES (?, ?)",
            ("openai_verbosity", value.value),
        )

    @staticmethod
    def get_openai_reasoning_effort(cursor: Cursor) -> OpenAIReasoningEffort:
        """Get the OpenAI reasoning effort configuration."""
        row = cursor.execute(
            "SELECT value FROM configuration WHERE key = ?",
            ("openai_reasoning_effort",),
        ).fetchone()
        value = row[0] if row else OpenAIReasoningEffort.MEDIUM.value
        return OpenAIReasoningEffort(value)

    @staticmethod
    def set_openai_reasoning_effort(
        cursor: Cursor, value: OpenAIReasoningEffort
    ) -> None:
        """Set the OpenAI reasoning effort configuration."""
        cursor.execute(
            "INSERT OR REPLACE INTO configuration (key, value) VALUES (?, ?)",
            ("openai_reasoning_effort", value.value),
        )

    @staticmethod
    def get_openai_service_tier(cursor: Cursor) -> OpenAIServiceTier:
        """Get the OpenAI service tier configuration."""
        row = cursor.execute(
            "SELECT value FROM configuration WHERE key = ?",
            ("openai_service_tier",),
        ).fetchone()
        value = row[0] if row else OpenAIServiceTier.DEFAULT.value
        return OpenAIServiceTier(value)

    @staticmethod
    def set_openai_service_tier(cursor: Cursor, value: OpenAIServiceTier) -> None:
        """Set the OpenAI service tier configuration."""
        cursor.execute(
            "INSERT OR REPLACE INTO configuration (key, value) VALUES (?, ?)",
            ("openai_service_tier", value.value),
        )

    @staticmethod
    def get_openai_timeout_minutes(cursor: Cursor) -> int:
        """Get the OpenAI timeout configuration in minutes."""
        row = cursor.execute(
            "SELECT value FROM configuration WHERE key = ?",
            ("openai_timeout_minutes",),
        ).fetchone()
        value = row[0] if row else "60"  # Default 60 minutes
        return int(value)

    @staticmethod
    def set_openai_timeout_minutes(cursor: Cursor, value: int) -> None:
        """Set the OpenAI timeout configuration in minutes."""
        cursor.execute(
            "INSERT OR REPLACE INTO configuration (key, value) VALUES (?, ?)",
            ("openai_timeout_minutes", str(value)),
        )

    @staticmethod
    def get_openai_compression_threshold(cursor: Cursor) -> int:
        """Get the OpenAI compression threshold configuration in tokens."""
        row = cursor.execute(
            "SELECT value FROM configuration WHERE key = ?",
            ("openai_compression_threshold",),
        ).fetchone()
        value = row[0] if row else "320000"  # Default 320,000 tokens
        return int(value)

    @staticmethod
    def set_openai_compression_threshold(cursor: Cursor, value: int) -> None:
        """Set the OpenAI compression threshold configuration in tokens."""
        cursor.execute(
            "INSERT OR REPLACE INTO configuration (key, value) VALUES (?, ?)",
            ("openai_compression_threshold", str(value)),
        )

    @staticmethod
    def _get_prompt(cursor: Cursor, key: str, filename: str) -> str:
        """Generic helper to get prompt configuration."""
        row = cursor.execute(
            "SELECT value FROM configuration WHERE key = ?", (key,)
        ).fetchone()
        if row:
            return str(row[0])
        # Load from file if not in database
        content: str = (files("bookwiki.data") / filename).read_text()
        return content

    @staticmethod
    def _set_prompt(cursor: Cursor, key: str, value: str) -> None:
        """Generic helper to set prompt configuration."""
        cursor.execute(
            "INSERT OR REPLACE INTO configuration (key, value) VALUES (?, ?)",
            (key, value),
        )

    @staticmethod
    def get_system_prompt(cursor: Cursor) -> str:
        """Get the system prompt configuration."""
        return Configuration._get_prompt(cursor, "system_prompt", "system_prompt.txt")

    @staticmethod
    def set_system_prompt(cursor: Cursor, value: str) -> None:
        """Set the system prompt configuration."""
        Configuration._set_prompt(cursor, "system_prompt", value)

    @staticmethod
    def get_chapter_prompt(cursor: Cursor) -> str:
        """Get the chapter prompt configuration."""
        return Configuration._get_prompt(cursor, "chapter_prompt", "chapter_prompt.txt")

    @staticmethod
    def set_chapter_prompt(cursor: Cursor, value: str) -> None:
        """Set the chapter prompt configuration."""
        Configuration._set_prompt(cursor, "chapter_prompt", value)

    @staticmethod
    def get_compress_prompt(cursor: Cursor) -> str:
        """Get the compress prompt configuration."""
        return Configuration._get_prompt(
            cursor, "compress_prompt", "compress_prompt.txt"
        )

    @staticmethod
    def set_compress_prompt(cursor: Cursor, value: str) -> None:
        """Set the compress prompt configuration."""
        Configuration._set_prompt(cursor, "compress_prompt", value)
