import logging
from typing import Annotated, Any, Self, final

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, model_validator
from pydantic.json_schema import SkipJsonSchema

from bookwiki.models import Block, Conversation

logger = logging.getLogger(__name__)


class LLMSolvableError(Exception):
    """Exception for errors that should be reported back to the LLM.

    This exception type indicates errors that the LLM should be made aware of
    and given a chance to resolve, rather than causing the system to crash.
    """

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class ToolModel(BaseModel):
    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")

    tool_id: Annotated[str, SkipJsonSchema(), Field(exclude=True)]
    tool_name: Annotated[str, SkipJsonSchema(), Field(exclude=True)]

    @classmethod
    def get_tool_description(cls) -> str:
        """Get the tool description from the class docstring."""
        return cls.__doc__ or ""

    @model_validator(mode="before")
    @classmethod
    def inject_from_context(cls, data: Any, info: ValidationInfo) -> Any:
        # Insert context values if missing from the payload
        if isinstance(data, dict):
            ctx = info.context or {}
            if "tool_id" not in data and "tool_id" in ctx:
                data = {**data, "tool_id": ctx["tool_id"]}
            if "tool_name" not in data and "tool_name" in ctx:
                data = {**data, "tool_name": ctx["tool_name"]}
        return data

    @model_validator(mode="after")
    def check_modelname_matches_class(self) -> Self:
        expected = type(self).__name__
        if self.tool_name != expected:
            raise ValueError(f'modelname must be "{expected}", got "{self.tool_name}"')
        return self

    def add_to_conversation(self, conv: Conversation) -> Block:
        """Add this tool to a conversation and return the created block.

        Args:
            conv: The Conversation object to add this tool to

        Returns:
            The Block that was created for this tool use
        """
        # Use Pydantic's model_dump_json() for direct JSON serialization
        params_json = self.model_dump_json()

        return conv.add_tool_use(
            name=self.__class__.__name__,
            use_id=self.tool_id,
            params=params_json,
        )

    @final
    def apply(self, block: Block) -> None:
        """Apply this tool in the context of a conversation.

        Args:
            block: The block that called this tool
        """
        logger.info(f"Executing tool {self.__class__.__name__} (id: {self.tool_id})")
        try:
            self._apply(block)
            logger.debug(f"Tool {self.__class__.__name__} completed successfully")
        except LLMSolvableError as e:
            logger.info(
                f"Tool {self.__class__.__name__} encountered LLM-solvable error: "
                f"{e.message}"
            )
            block.respond_error(e.message)

    def _apply(self, block: Block) -> None:
        raise NotImplementedError(
            f"apply() not implemented for {self.__class__.__name__}"
        )
