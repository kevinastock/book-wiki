import logging

from bookwiki.models import Block, Prompt
from bookwiki.tools.base import LLMSolvableError, ToolModel

logger = logging.getLogger(__name__)


class RequestExpertFeedback(ToolModel):
    """Request help from a human expert when you encounter uncertain situations or
    need guidance.

    Use this tool when you have meaningful uncertainty, particularly for decisions
    that could create cascading problems in future chapters.
    Common situations include:
    - Character identities or relationships that seem unclear from the text
    - Narrative structures that don't fit standard wiki organization
    - Information that might be conflicting or ambiguous
    - Editorial decisions that could affect multiple pages or future processing
    - Any situation where you're making assumptions that could compound over time
    - Quality concerns about patterns you're establishing
    - When you sense something might be "off" but aren't sure what

    Frame your request clearly and provide context about what you're trying to
    resolve. It's better to ask for guidance on potentially impactful decisions
    than to discover systemic problems after they've affected multiple chapters."""

    request: str
    """Clear, specific request for help with sufficient context about the situation.
    Explain what you've tried, what's unclear, and what kind of guidance
    would help you proceed. Use markdown formatting if needed, but generally
    keep it simple - links are not supported in feedback."""

    def _apply(self, block: Block) -> None:  # noqa: ARG002
        # This is just a no-op here; the newly created block will be found by
        # another piece of code which handles getting user input.
        logger.info(
            f"Feedback requested: {self.request[:100]}..."
            if len(self.request) > 100
            else f"Feedback requested: {self.request}"
        )
        pass


class SpawnAgent(ToolModel):
    """Create a focused subagent to handle a specific task using a stored prompt
    template.

    Use this tool to break down complex work into focused subtasks, especially when:
    - Processing individual entities within a chapter (one subagent per
      character/location/concept)
    - Reading and summarizing multiple wiki pages with a specific question in mind
    - Handling complex related updates across multiple pages
    - Investigating specific relationships or connections

    The subagent will have access to all the same tools and can spawn its own
    subagents when needed. Provide clear, specific variable values that give the
    subagent enough context to complete its task.
    """

    prompt_key: str
    """The identifier of the prompt template to use (from ListPrompts).
    Use ShowPrompt first to see what variables the template requires."""

    template_names: list[str]
    """List of variable names to substitute in the prompt template.
    Variable names should NOT include the $ sign.

    Example: If the template contains "$chapter_number" and "$character_name", provide:
    ["chapter_number", "character_name"]"""

    template_values: list[str]
    """List of values corresponding to the variable names in the same order.
    Must be the same length as template_names.

    Example: For variables ["chapter_number", "character_name"], provide:
    ["5", "Gandalf"]"""

    def _apply(
        self,
        block: Block,
    ) -> None:
        cursor = block.get_cursor()

        # Validate that template_names and template_values have the same length
        if len(self.template_names) != len(self.template_values):
            raise LLMSolvableError(
                f"Variable names and values lists must have the same length. "
                f"Got {len(self.template_names)} names and "
                f"{len(self.template_values)} values."
            )

        # Check for $ signs in variable names
        for name in self.template_names:
            if "$" in name:
                raise LLMSolvableError(
                    f"Variable name '{name}' contains '$' character. "
                    f"Variable names should not include the '$' prefix."
                )

        # Convert to dictionary for template substitution
        template_vars = dict(
            zip(self.template_names, self.template_values, strict=False)
        )

        logger.info(
            f"Spawning agent with prompt '{self.prompt_key}' and vars: "
            f"{list(template_vars.keys())}"
        )
        prompt = Prompt.get_prompt(cursor, self.prompt_key)

        if prompt is None:
            raise LLMSolvableError("No prompt with that key exists")

        expected_keys = sorted(prompt.template.get_identifiers())
        actual_keys = sorted(template_vars.keys())
        if expected_keys != actual_keys:
            raise LLMSolvableError(
                f"Failed to substitute vars!\n"
                f"Expected keys: {expected_keys}\nActual keys: {actual_keys}"
            )

        text = prompt.template.substitute(template_vars)

        new_conversation = block.start_conversation()
        new_conversation.add_user_text(text)
