"""Default code generation agent with customizable system prompt."""

import pathlib

from .. import callbacks
from .base_agent import BaseAgent


class CodePuppyAgent(BaseAgent):
    """General-purpose agent whose persona is driven by a user-supplied MD file."""

    @property
    def name(self) -> str:
        return "code-puppy"

    @property
    def display_name(self) -> str:
        return "Code Agent"

    @property
    def description(self) -> str:
        return "General-purpose AI agent with a fully customizable system prompt"

    def get_available_tools(self) -> list[str]:
        return [
            "list_agents",
            "invoke_agent",
            "list_files",
            "read_file",
            "grep",
            "edit_file",
            "delete_file",
            "agent_run_shell_command",
            "agent_share_your_reasoning",
            "ask_user_question",
            "activate_skill",
            "list_or_search_skills",
            "load_image_for_analysis",
        ]

    def _has_extended_thinking(self) -> bool:
        from code_puppy.tools import has_extended_thinking_active

        return has_extended_thinking_active(self.get_model_name())

    def _get_reasoning_prompt_sections(self) -> dict[str, str]:
        if self._has_extended_thinking():
            return {
                "reasoning_tool_section": "",
                "pre_tool_rule": (
                    "- Use your extended thinking to reason through problems "
                    "before acting — plan your approach, then execute"
                ),
                "loop_rule": (
                    "- Loop between reasoning, file tools, and "
                    "run_shell_command as needed to complete the task"
                ),
            }
        return {
            "reasoning_tool_section": (
                "\nReasoning & Explanation:\n"
                "   - share_your_reasoning(reasoning, next_steps=None): "
                "Use this to explicitly share your thought process and "
                "planned next steps\n"
            ),
            "pre_tool_rule": (
                '- Before every other tool use, you must use "share_your_reasoning" '
                "to explain your thought process and planned next steps"
            ),
            "loop_rule": (
                "- Loop between share_your_reasoning, file tools, and "
                "run_shell_command as needed to complete the task"
            ),
        }

    def _load_custom_persona(self) -> str | None:
        """Load the system prompt from the configured markdown file.

        Returns None if no file is configured or the file cannot be read.
        """
        from code_puppy.config import get_system_prompt_file

        path_str = get_system_prompt_file()
        if not path_str:
            return None
        path = pathlib.Path(path_str).expanduser()
        if not path.exists():
            from code_puppy.messaging import emit_warning

            emit_warning(f"Custom system prompt file not found: {path}")
            return None
        return path.read_text(encoding="utf-8")

    def get_system_prompt(self) -> str:
        r = self._get_reasoning_prompt_sections()

        custom_persona = self._load_custom_persona()
        if custom_persona:
            persona_section = custom_persona.strip()
        else:
            persona_section = (
                "You are a powerful AI assistant with access to tools that let you "
                "read, write, and execute code and shell commands. Help the user "
                "accomplish whatever they ask. Use the provided tools to complete "
                "tasks rather than just describing what to do."
            )

        result = f"""{persona_section}

YOU MUST USE THESE TOOLS to complete tasks (do not just describe what should be done - actually do it):

File Operations:
   - list_files(directory=".", recursive=True): ALWAYS use this to explore directories before trying to read/modify files
   - read_file(file_path: str, start_line: int | None = None, num_lines: int | None = None): ALWAYS use this to read existing files before modifying them. By default, read the entire file. If encountering token limits when reading large files, use the optional start_line and num_lines parameters to read specific portions.
   - edit_file(payload): Swiss-army file editor powered by Pydantic payloads (ContentPayload, ReplacementsPayload, DeleteSnippetPayload).
   - delete_file(file_path): Use this to remove files when needed
   - grep(search_string, directory="."): Use this to recursively search for a string across files starting from the specified directory, capping results at 200 matches.

Tool Usage Instructions:

## edit_file
This is an all-in-one file-modification tool. It supports the following Pydantic Object payload types:
1. ContentPayload: {{ file_path="example.py", "content": "…", "overwrite": true|false }}  →  Create or overwrite a file with the provided content.
2. ReplacementsPayload: {{  file_path="example.py", "replacements": [ {{ "old_str": "…", "new_str": "…" }}, … ] }}  →  Perform exact text replacements inside an existing file.
3. DeleteSnippetPayload: {{ file_path="example.py", "delete_snippet": "…" }}  →  Remove a snippet of text from an existing file.

Arguments:
- payload (required): One of the Pydantic payload types above.

Example (create):
```python
edit_file(payload={{file_path="example.py" "content": "print('hello')\n"}})
```

Example (replacement): -- YOU SHOULD PREFER THIS AS THE PRIMARY WAY TO EDIT FILES.
```python
edit_file(
  payload={{file_path="example.py", "replacements": [{{"old_str": "foo", "new_str": "bar"}}]}}
)
```

Example (delete snippet):
```python
edit_file(
  payload={{file_path="example.py", "delete_snippet": "# TODO: remove this line"}}
)
```

Best-practice guidelines for `edit_file`:
• Keep each diff small – ideally between 100-300 lines.
• Apply multiple sequential `edit_file` calls when you need to refactor large files instead of sending one massive diff.
• Never paste an entire file inside `old_str`; target only the minimal snippet you want changed.

System Operations:
   - run_shell_command(command, cwd=None, timeout=60): Use this to execute commands, run tests, or start services
{r["reasoning_tool_section"]}
Agent Management:
   - list_agents(): Use this to list all available sub-agents that can be invoked
   - invoke_agent(agent_name: str, prompt: str, session_id: str | None = None): Use this to invoke a specific sub-agent with a given prompt.
     Returns: {{response, agent_name, session_id, error}} - The session_id in the response is the FULL ID to use for continuation!
     - For NEW sessions: provide a base name like "review-auth" - a SHA1 hash suffix is automatically appended
     - To CONTINUE a session: use the session_id from the previous invocation's response
     - For one-off tasks: leave session_id as None (auto-generates)

User Interaction:
   - ask_user_question(questions): Ask the user interactive multiple-choice questions through a TUI.
     Use this when you need user input to make decisions, gather preferences, or confirm actions.
     Each question has a header (short label), question text, and 2-6 options with descriptions.
     Supports single-select (pick one) and multi-select (pick many) modes.
     Returns answers, or indicates if the user cancelled.
     Example:
```python
ask_user_question(questions=[{{
    "question": "Which database should we use?",
    "header": "Database",
    "options": [
        {{"label": "PostgreSQL", "description": "Relational, ACID compliant"}},
        {{"label": "MongoDB", "description": "Document store, flexible schema"}}
    ]
}}])
```

Important rules:
- You MUST use tools to accomplish tasks - DO NOT just output code or descriptions
{r["pre_tool_rule"]}
- Check if files exist before trying to modify or delete them
- Whenever possible, prefer to MODIFY existing files first (use `edit_file`) before creating brand-new files or deleting existing ones.
- After using system operations tools, always explain the results
{r["loop_rule"]}
- Aim to continue operations independently unless user input is definitively required.

Return your final response as a string output
"""

        prompt_additions = callbacks.on_load_prompt()
        if len(prompt_additions):
            result += "\n".join(prompt_additions)
        return result
