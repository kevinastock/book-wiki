"""LLM service interface and types for dependency inversion."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable

from bookwiki.models import Block
from bookwiki.tools import ToolModel


class LLMRetryableError(Exception):
    """Exception for retryable LLM errors (rate limits, server issues, etc)."""

    pass


class LLMNonRetryableError(Exception):
    """Exception for non-retryable LLM errors (invalid requests, auth issues, etc)."""

    pass


@dataclass(frozen=True)
class LLMResponse:
    """Encapsulates the response from the LLM.

    This type hides the provider-specific Message type and provides
    a clean interface for the rest of the codebase to interact with
    LLM responses.
    """

    tools: list[ToolModel]
    texts: list[str]
    updated_prev: str
    compressing: bool

    # Token usage information
    input_tokens: int
    output_tokens: int


class LLMService(ABC):
    """Abstract interface for LLM providers."""

    @abstractmethod
    def prompt(
        self,
        previously: str | None,
        new_messages: Iterable[Block],
        *,
        system_message: str = "",
        compressing: bool = False,
    ) -> str:
        """Submit a prompt to the LLM and return a response ID for later retrieval."""
        pass

    @abstractmethod
    def try_fetch(self, response_id: str) -> LLMResponse | None:
        pass

    @abstractmethod
    def get_compression_threshold(self) -> int:
        """Get the token count threshold for triggering conversation compression.

        Returns the number of tokens at which a conversation should be compressed.
        """
        pass
