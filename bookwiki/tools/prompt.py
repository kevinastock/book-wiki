from string import Template

from bookwiki.models import Block, Prompt
from bookwiki.tools.base import LLMSolvableError, ToolModel


class ListPrompts(ToolModel):
    """Display all available prompt templates that can be used with the SpawnAgent tool.

    Use this to discover what specialized prompt templates are available for creating
    focused subagents. Each prompt has a key identifier and summary describing its
    purpose."""

    def _apply(
        self,
        block: Block,
    ) -> None:
        prompts = Prompt.list_prompts(block.get_cursor())

        if not prompts:
            block.respond("There are no stored prompts.")
        else:
            responses = []
            for _, p in sorted(prompts.items()):
                variables = sorted(p.template.get_identifiers())
                var_str = f"Variables: {variables}" if variables else "Variables: none"
                responses.append(f"Key: {p.key}\nSummary: {p.summary}\n{var_str}\n")
            block.respond("\n".join(responses))


class ShowPrompt(ToolModel):
    """Retrieve the full details of a specific prompt template including its template
    text and variable placeholders.

    Use this to examine a prompt template before using it with SpawnAgent, or to
    understand what variables need to be provided when spawning an agent with this
    prompt."""

    key: str
    """The unique identifier for the prompt template (as shown in ListPrompts
    results)."""

    def _apply(
        self,
        block: Block,
    ) -> None:
        prompts = Prompt.list_prompts(block.get_cursor())

        if self.key not in prompts:
            raise LLMSolvableError(f"Key {self.key} does not exist.")

        p = prompts[self.key]
        variables = sorted(p.template.get_identifiers())
        var_str = f"Variables: {variables}" if variables else "Variables: none"
        block.respond(
            f"Summary: {p.summary}\n{var_str}\nTemplate: {p.template.template}"
        )


class WritePrompt(ToolModel):
    """Create and store a new prompt template for future use with the SpawnAgent tool.

    CRITICAL: Subagents receive ONLY the system prompt, tools, and your custom prompt.
    They have NO access to conversation history or context. Your prompts must be
    completely self-contained and include all necessary information.

    Templates use Python string template syntax with $variable placeholders that will
    be substituted when the prompt is used. Follow the prompt writing guidelines in
    the system prompt to create effective, autonomous subagent instructions."""

    key: str
    """Unique identifier for this prompt template (e.g., 'character-analyzer',
    'location-summarizer'). Use descriptive names that indicate the prompt's purpose."""

    summary: str
    """Brief description of what this prompt template does and when to use it
    (e.g., 'Analyzes a character and updates their wiki page with new information')."""

    template: str
    """The complete prompt text with $variable placeholders (e.g., $chapter_number,
    $character_name, $entity_type). Variables must use the dollar sign syntax:
    $variable_name. Must include all context, instructions, and expected deliverables
    since subagents have no access to your conversation history. Follow the system
    prompt's template structure.

    Example variable usage:
    - "Process the current chapter for character `$character_names`"
    - "Update wiki page for '$entity_type' named '$entity_name'"
    - "Analyze relationships in the current chapter involving '$character_list'"

    DO NOT do anything like this:
    - "$entity_name: The exact name of the entity"
    This will be very confusing as the agent will see something like
    - "Earth: The exact name of the entity"
    Instead, DO THIS:
    - "The exact name of the entity: '$entity_name'"
    """

    def _apply(
        self,
        block: Block,
    ) -> None:
        template = Template(self.template)
        if not template.is_valid():
            raise LLMSolvableError("Template is not valid, prompt rejected.")

        block.add_prompt(self.key, self.summary, template)
        block.respond("Prompt stored")
