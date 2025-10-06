"""Configuration enums for the bookwiki application."""

from enum import Enum


class OpenAIModel(Enum):
    """OpenAI model options."""

    GPT_5 = "gpt-5"
    GPT_5_MINI = "gpt-5-mini"
    GPT_5_NANO = "gpt-5-nano"


class OpenAIVerbosity(Enum):
    """OpenAI verbosity options."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class OpenAIReasoningEffort(Enum):
    """OpenAI reasoning effort options."""

    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class OpenAIServiceTier(Enum):
    """OpenAI service tier options."""

    FLEX = "flex"
    DEFAULT = "default"
