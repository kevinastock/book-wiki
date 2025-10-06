import functools
import logging
import time
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Literal, Tuple, Type, TypeVar

import openai
from dotenv import load_dotenv
from openai import OpenAI
from openai._types import NOT_GIVEN, NotGiven
from openai.types.responses import (
    EasyInputMessageParam,
    FunctionToolParam,
    ResponseFunctionToolCall,
    ResponseInputParam,
    ResponseTextConfigParam,
)
from openai.types.responses.response_input_param import FunctionCallOutput
from openai.types.shared_params import Reasoning
from pydantic import ValidationError
from pydantic.json_schema import GenerateJsonSchema

from bookwiki.config_enums import (
    OpenAIModel,
    OpenAIReasoningEffort,
    OpenAIServiceTier,
    OpenAIVerbosity,
)
from bookwiki.llm import (
    LLMNonRetryableError,
    LLMResponse,
    LLMRetryableError,
    LLMService,
)
from bookwiki.models import Block
from bookwiki.tools import deserialize_tool
from bookwiki.tools.base import ToolModel

T = TypeVar("T")

load_dotenv()
logger = logging.getLogger(__name__)

# TODO: add to this as we discover errors that belong here.
RETRYABLE_CODES: list[str] = ["server_error"]

COMPRESS_KEY = "compress"
COMPRESS_VALUE = "true"


def retry_with_backoff(
    exception_types: list[Type[Exception]],
    max_attempts: int = 5,
    base_delay: float = 5.0,
    max_delay: float = 100.0,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for retrying functions with exponential backoff.

    Args:
        exception_types: List of exception types to retry on
        max_attempts: Maximum number of attempts (default: 5)
        base_delay: Initial delay in seconds (default: 5.0)
        max_delay: Maximum delay in seconds (default: 100.0)
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            delay = base_delay
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    # Check if this exception type should be retried
                    if not any(isinstance(e, exc_type) for exc_type in exception_types):
                        raise

                    last_exception = e

                    if attempt < max_attempts - 1:
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_attempts} failed for "
                            f"{func.__name__}: {e}. Retrying in {delay} seconds..."
                        )
                        time.sleep(delay)
                        delay = min(
                            delay * 2, max_delay
                        )  # Exponential backoff with cap
                    else:
                        logger.error(
                            f"All {max_attempts} attempts failed for "
                            f"{func.__name__}: {e}"
                        )

            # All attempts exhausted
            if last_exception:
                raise last_exception
            raise RuntimeError(f"Unexpected error in retry logic for {func.__name__}")

        return wrapper

    return decorator


class GenerateJsonSchemaNoTitles(GenerateJsonSchema):
    def field_title_should_be_set(self, _: Any) -> bool:
        return False

    def _update_class_schema(self, json_schema: Any, cls: Any, config: Any) -> None:
        super()._update_class_schema(json_schema, cls, config)
        json_schema.pop("title", None)
        json_schema.pop("description", None)

        # For OpenAI strict mode, all properties must be in the required array
        # even if they can be null (optional fields use anyOf with null type)
        if "properties" in json_schema:
            json_schema["required"] = list(json_schema["properties"].keys())


class OpenAILLMService(LLMService):
    def __init__(
        self,
        *,
        model: OpenAIModel,
        service_tier: OpenAIServiceTier,
        tools: Tuple[Type[ToolModel], ...],
        system_message: str,
        compression_threshold: int,
        verbosity: OpenAIVerbosity,
        reasoning_effort: OpenAIReasoningEffort,
        timeout_minutes: int,
    ) -> None:
        self.client = OpenAI()
        self.model = model
        self.tools = tools
        self.service_tier = service_tier
        self.system_message = system_message
        self.compression_threshold = compression_threshold
        self.verbosity = verbosity
        self.reasoning_effort = reasoning_effort
        self.timeout_minutes = timeout_minutes
        self._tool_params: list[FunctionToolParam] | NotGiven = (
            self._create_tools_from_models(tools) if tools else NOT_GIVEN
        )
        self._tool_choice: Literal["auto"] | NotGiven = "auto" if tools else NOT_GIVEN

    def set_model(self, model: OpenAIModel) -> None:
        """Update the OpenAI model configuration."""
        self.model = model

    def set_service_tier(self, service_tier: OpenAIServiceTier) -> None:
        """Update the OpenAI service tier configuration."""
        self.service_tier = service_tier

    def set_verbosity(self, verbosity: OpenAIVerbosity) -> None:
        """Update the OpenAI verbosity configuration."""
        self.verbosity = verbosity

    def set_reasoning_effort(self, reasoning_effort: OpenAIReasoningEffort) -> None:
        """Update the OpenAI reasoning effort configuration."""
        self.reasoning_effort = reasoning_effort

    def set_system_message(self, system_message: str) -> None:
        """Update the system message."""
        self.system_message = system_message

    def set_compression_threshold(self, compression_threshold: int) -> None:
        """Update the compression threshold."""
        self.compression_threshold = compression_threshold

    def set_timeout_minutes(self, timeout_minutes: int) -> None:
        """Update the timeout in minutes."""
        self.timeout_minutes = timeout_minutes

    @staticmethod
    def _convert_blocks_to_input(blocks: Iterable[Block]) -> ResponseInputParam:
        """Convert v2 Block messages to OpenAI input format."""
        input_items: ResponseInputParam = []

        for block in blocks:
            if block.text_body is not None:
                # User text message
                assert block.text_role == "user"
                message: EasyInputMessageParam = {
                    "type": "message",
                    "role": block.text_role,  # type: ignore
                    "content": block.text_body,
                }
                input_items.append(message)
            elif block.tool_response is not None:
                # Tool response if available
                assert block.tool_use_id is not None
                output: FunctionCallOutput = {
                    "type": "function_call_output",
                    "call_id": block.tool_use_id,
                    "output": block.tool_response,
                }
                input_items.append(output)
            else:
                raise ValueError(
                    f"Block must have either text_body or tool_response, got: {block}"
                )

        return input_items

    @staticmethod
    def _create_tools_from_models(
        tool_models: Tuple[Type[ToolModel], ...],
    ) -> list[FunctionToolParam]:
        """Convert ToolModel classes to OpenAI function tool parameters."""
        tools: list[FunctionToolParam] = []

        for tool_model in tool_models:
            tool: FunctionToolParam = {
                "type": "function",
                "name": tool_model.__name__,
                "description": tool_model.get_tool_description(),
                "parameters": tool_model.model_json_schema(
                    schema_generator=GenerateJsonSchemaNoTitles
                ),
                "strict": True,
            }
            tools.append(tool)

        return tools

    def _parse_tool_models(
        self,
        tool_call: ResponseFunctionToolCall,
    ) -> ToolModel:
        """Parse OpenAI tool calls into ToolModel instances."""
        assert tool_call.id is not None

        try:
            return deserialize_tool(
                tool_call.arguments,
                self.tools,
                tool_id=tool_call.call_id,
                tool_name=tool_call.name,
            )
        except ValidationError as e:
            raise LLMRetryableError(f"Failed to parse tool call: {e}") from e

    @retry_with_backoff(
        exception_types=[
            openai.APIConnectionError,
            openai.APITimeoutError,
            openai.RateLimitError,
            openai.InternalServerError,
        ]
    )
    def prompt(
        self,
        previously: str | None,
        new_messages: Iterable[Block],
        *,
        system_message: str = "",
        compressing: bool = False,
    ) -> str:
        """Submit a prompt to the OpenAI API and return the response."""
        logger.info(
            f"Sending request to OpenAI API with model: {self.model.value}, "
            f"service_tier: {self.service_tier.value}"
        )

        # Debug logging of block contents
        block_list = list(new_messages)
        logger.debug(f"Converting {len(block_list)} blocks to input")
        logger.debug(f"Previous response ID: {previously}")
        for i, block in enumerate(block_list):
            text_info = f"[{len(block.text_body)} chars]" if block.text_body else None
            tool_resp_info = (
                f"[{len(block.tool_response)} chars]" if block.tool_response else None
            )
            logger.debug(
                f"Block {i}: id={block.id}, text_role={block.text_role}, "
                f"text_body={text_info}, tool_use_id={block.tool_use_id}, "
                f"tool_response={tool_resp_info}"
            )
            if block.text_body:
                logger.debug(f"Block {i} text content: {block.text_body[:100]}...")
            if block.tool_response:
                logger.debug(f"Block {i} tool response: {block.tool_response}")

        new_messages = block_list  # Convert back to iterable

        # Build reasoning config if needed
        reasoning_config: Reasoning = {
            "effort": self.reasoning_effort.value,
        }

        text_config: ResponseTextConfigParam = {"verbosity": self.verbosity.value}

        response = self.client.responses.create(
            model=self.model.value,
            input=OpenAILLMService._convert_blocks_to_input(new_messages),
            background=True,
            service_tier=self.service_tier.value,
            instructions=system_message or self.system_message,
            tools=() if compressing else self._tool_params,
            tool_choice=self._tool_choice,
            previous_response_id=previously if previously else NOT_GIVEN,
            reasoning=reasoning_config,
            text=text_config,
            metadata={COMPRESS_KEY: COMPRESS_VALUE if compressing else ""},
        )

        return response.id

    @retry_with_backoff(
        exception_types=[
            openai.APIConnectionError,
            openai.APITimeoutError,
            openai.RateLimitError,
            openai.InternalServerError,
        ]
    )
    def try_fetch(self, response_id: str) -> LLMResponse | None:
        response = self.client.responses.retrieve(response_id)
        if response.status in {"queued", "in_progress"}:
            # Check if response has timed out
            created_at = datetime.fromtimestamp(response.created_at, tz=timezone.utc)
            now = datetime.now(tz=timezone.utc)
            elapsed_minutes = (now - created_at).total_seconds() / 60

            if elapsed_minutes > self.timeout_minutes:
                logger.warning(
                    f"Response {response_id} timed out after {elapsed_minutes:.1f} "
                    f"minutes (limit: {self.timeout_minutes} minutes). Cancelling..."
                )
                try:
                    self.client.responses.cancel(response_id)
                    logger.info(f"Successfully cancelled response {response_id}")
                except Exception as e:
                    logger.error(f"Failed to cancel response {response_id}: {e}")

                raise LLMRetryableError(
                    f"Response timed out after {self.timeout_minutes} minutes"
                )

            return None

        if response.status == "failed":
            assert response.error is not None

            message = f"Failed! {response.error.code}: {response.error.message}"
            if response.error.code in RETRYABLE_CODES:
                raise LLMRetryableError(message)
            raise LLMNonRetryableError(message)
        elif response.status == "cancelled":
            raise LLMRetryableError("Response was cancelled")
        elif response.status != "completed":
            raise LLMNonRetryableError(f"Unknown response status: {response.status}")

        # Parse the response
        tools_used = []
        texts = []

        for output_item in response.output:
            if output_item.type == "message":
                # Process content which is a list of text/refusal items
                for content_item in output_item.content:
                    if content_item.type == "output_text":
                        texts.append(content_item.text)
                    elif content_item.type == "refusal":
                        raise LLMNonRetryableError(f"Refusal: {content_item.refusal}")
                    else:
                        raise LLMNonRetryableError(
                            f"Unknown content type: {content_item.type}"
                        )
            elif output_item.type == "function_call":
                # Parse tool calls
                tool_instance = self._parse_tool_models(output_item)
                tools_used.append(tool_instance)

        # Get token usage from the response
        input_tokens = response.usage.input_tokens if response.usage else 0
        output_tokens = response.usage.output_tokens if response.usage else 0

        compressing = False
        if (
            response.metadata
            and response.metadata.get(COMPRESS_KEY, None) == COMPRESS_VALUE
        ):
            compressing = True

        return LLMResponse(
            tools=tools_used,
            texts=texts,
            updated_prev=response.id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            compressing=compressing,
        )

    def get_compression_threshold(self) -> int:
        return self.compression_threshold
