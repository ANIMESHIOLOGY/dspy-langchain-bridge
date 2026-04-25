"""LangChain Memory → DSPy history bridge.

Adapts LangChain conversation-memory objects (``ConversationBufferMemory``,
``ConversationBufferWindowMemory``, ``ConversationSummaryMemory``, …) into the
turn-list format that DSPy multi-turn modules expect as ``history``.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# DSPy history format
# ---------------------------------------------------------------------------
# DSPy multi-turn modules conventionally receive history as a list of dicts:
#   [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
# This matches the OpenAI chat messages convention.

HistoryEntry = dict[str, str]


class LangChainMemoryAdapter:
    """Bridges a LangChain memory object into DSPy's history format.

    Reads the conversation turns stored in a LangChain memory object and
    exposes them as a plain list of ``{"role": ..., "content": ...}`` dicts
    that can be passed as ``history=`` to any DSPy multi-turn module.

    It also provides ``save_turn`` to write a new user/assistant turn back
    into the LangChain memory so both sides stay in sync.

    Args:
        memory: A LangChain memory instance (e.g. ``ConversationBufferMemory``).
        human_role: Role label to use for human turns. Defaults to ``"user"``.
        ai_role: Role label to use for AI turns. Defaults to ``"assistant"``.

    Example:
        >>> from langchain.memory import ConversationBufferMemory
        >>> from dspy_lc_bridge import LangChainMemoryAdapter
        >>>
        >>> mem = ConversationBufferMemory(return_messages=True)
        >>> adapter = LangChainMemoryAdapter(mem)
        >>>
        >>> class ChatBot(dspy.Module):
        ...     def __init__(self):
        ...         self.respond = dspy.ChainOfThought("history, question -> answer")
        ...
        ...     def forward(self, question):
        ...         history = adapter.get_history()
        ...         pred = self.respond(history=history, question=question)
        ...         adapter.save_turn(question, pred.answer)
        ...         return pred
    """

    def __init__(
        self,
        memory: Any,
        human_role: str = "user",
        ai_role: str = "assistant",
    ) -> None:
        self.memory = memory
        self.human_role = human_role
        self.ai_role = ai_role

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_history(self) -> list[HistoryEntry]:
        """Return conversation history as a list of role/content dicts.

        Returns:
            List of ``{"role": "user"|"assistant", "content": "..."}`` dicts
            in chronological order.
        """
        messages = self._extract_messages()
        return self._messages_to_history(messages)

    def save_turn(self, human_input: str, ai_output: str) -> None:
        """Persist a completed conversation turn back into the LangChain memory.

        Args:
            human_input: The user's message for this turn.
            ai_output: The assistant's response for this turn.
        """
        if hasattr(self.memory, "save_context"):
            self.memory.save_context(
                {"input": human_input},
                {"output": ai_output},
            )

    def clear(self) -> None:
        """Clear all conversation history from the underlying LangChain memory."""
        if hasattr(self.memory, "clear"):
            self.memory.clear()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _extract_messages(self) -> list[Any]:
        """Extract LangChain message objects from the memory."""
        # return_messages=True path — memory stores Message objects directly
        if hasattr(self.memory, "chat_memory"):
            chat_memory = self.memory.chat_memory
            if hasattr(chat_memory, "messages"):
                return list(chat_memory.messages)

        # String-based memory (return_messages=False) — parse the history string
        if hasattr(self.memory, "load_memory_variables"):
            variables = self.memory.load_memory_variables({})
            # Common keys used by LangChain memory types
            for key in ("history", "chat_history"):
                val = variables.get(key)
                if val is not None:
                    if isinstance(val, list):
                        return val
                    # Fallback: raw string — wrap as a single human message
                    try:
                        from langchain_core.messages import HumanMessage

                        return [HumanMessage(content=str(val))]
                    except ImportError:
                        return []
        return []

    def _messages_to_history(self, messages: list[Any]) -> list[HistoryEntry]:
        """Convert LangChain Message objects to DSPy history dicts."""
        history: list[HistoryEntry] = []
        for msg in messages:
            content = msg.content if hasattr(msg, "content") else str(msg)
            msg_type = type(msg).__name__.lower()
            if "human" in msg_type or "user" in msg_type:
                role = self.human_role
            elif "ai" in msg_type or "assistant" in msg_type:
                role = self.ai_role
            elif "system" in msg_type:
                role = "system"
            else:
                role = "user"
            history.append({"role": role, "content": content})
        return history

    def __repr__(self) -> str:
        return (
            f"LangChainMemoryAdapter(memory={self.memory.__class__.__name__})"
        )
