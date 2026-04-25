"""LangChain LLM → DSPy LM backend bridge.

Lets DSPy use *any* LangChain-compatible LLM (``BaseLLM`` or
``BaseChatModel``) as its language-model backend, avoiding the need for a
separate litellm / API key configuration when you already have a LangChain LLM
set up.
"""

from __future__ import annotations

from typing import Any

from dspy_lc_bridge._types import ChatMessage


def _to_lc_messages(messages: list[ChatMessage]) -> list[Any]:
    """Convert OpenAI-style message dicts to LangChain message objects."""
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    lc: list[Any] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            lc.append(SystemMessage(content=content))
        elif role == "assistant":
            lc.append(AIMessage(content=content))
        else:
            lc.append(HumanMessage(content=content))
    return lc


def _extract_text(response: Any) -> str:
    """Extract the completion text from a LangChain LLM response."""
    if hasattr(response, "content"):
        return str(response.content)
    if hasattr(response, "text"):
        return str(response.text)
    return str(response)


class LangChainLM:
    """Wraps a LangChain LLM as a DSPy-compatible LM backend.

    Pass an instance to ``dspy.settings.configure(lm=...)`` so that DSPy
    modules call through to your LangChain LLM instead of requiring a
    separate litellm model string.

    Works with both ``BaseLLM`` (completion models) and ``BaseChatModel``
    (chat models).  The class auto-detects which interface to use.

    Args:
        llm: Any LangChain ``BaseLLM`` or ``BaseChatModel`` instance.
        temperature: Optional temperature override forwarded on every call.
        max_tokens: Optional max_tokens override.

    Example:
        >>> from langchain_openai import ChatOpenAI
        >>> from dspy_lc_bridge import LangChainLM
        >>> import dspy
        >>>
        >>> llm = ChatOpenAI(model="gpt-4o")
        >>> dspy.settings.configure(lm=LangChainLM(llm))
        >>> # Now all dspy.Module calls route through ChatOpenAI
    """

    def __init__(
        self,
        llm: Any,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self.llm = llm
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.history: list[dict[str, Any]] = []
        # Detect whether this is a chat model or a plain completion model
        self._is_chat = self._detect_chat_model(llm)

    @staticmethod
    def _detect_chat_model(llm: Any) -> bool:
        try:
            from langchain_core.language_models import BaseChatModel

            return isinstance(llm, BaseChatModel)
        except ImportError:
            return hasattr(llm, "invoke") and not hasattr(llm, "predict")

    def _build_invoke_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        return kwargs

    # ------------------------------------------------------------------
    # DSPy LM protocol
    # ------------------------------------------------------------------

    def __call__(
        self,
        prompt: str | None = None,
        messages: list[ChatMessage] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Call the LangChain LLM and return a list of completion strings.

        DSPy calls this method with either ``prompt`` (a raw string) or
        ``messages`` (a list of OpenAI-style dicts).  Both forms are
        supported.

        Args:
            prompt: Raw completion prompt (used by older DSPy modules).
            messages: List of ``{"role": ..., "content": ...}`` dicts.
            **kwargs: Extra kwargs forwarded to the LangChain LLM.

        Returns:
            A list containing a single completion string.  DSPy expects a
            list even when only one completion is returned.
        """
        if messages is None and prompt is not None:
            messages = [{"role": "user", "content": prompt}]
        elif messages is None:
            raise ValueError("Either 'prompt' or 'messages' must be provided.")

        invoke_kwargs = {**self._build_invoke_kwargs(), **kwargs}

        if self._is_chat:
            lc_messages = _to_lc_messages(messages)
            response = self.llm.invoke(lc_messages, **invoke_kwargs)
        else:
            # Plain completion model — concatenate messages into a single prompt
            full_prompt = "\n".join(m.get("content", "") for m in messages)
            response = self.llm.invoke(full_prompt, **invoke_kwargs)

        output = _extract_text(response)
        self.history.append({"messages": messages, "outputs": [output]})
        return [output]

    def inspect_history(self, n: int = 1) -> list[dict[str, Any]]:
        """Return the last *n* LM call records.

        Args:
            n: Number of history entries to return (most recent first).

        Returns:
            List of dicts with ``messages`` and ``outputs`` keys.
        """
        return self.history[-n:]

    def get_usage_metadata(self) -> dict[str, int]:
        """Return aggregated token usage metadata.

        Returns:
            Dict with ``total_calls`` key.  Extend this if your LangChain LLM
            surfaces per-call token counts.
        """
        return {"total_calls": len(self.history)}

    def __repr__(self) -> str:
        return f"LangChainLM(llm={self.llm.__class__.__name__}, is_chat={self._is_chat})"
