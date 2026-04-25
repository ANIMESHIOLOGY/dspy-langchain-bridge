"""Agent & tool interop bridges.

Bridge 7: ``DSPyTool``   — wrap a DSPy module as a LangChain ``BaseTool``.
Bridge 8: ``LangChainTool`` — use a LangChain tool inside DSPy agent loops.
Bridge 9: ``DSPyNode``   — wrap a DSPy module as a LangGraph node function.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Bridge 7: DSPy Module → LangChain BaseTool
# ---------------------------------------------------------------------------


class DSPyTool:
    """Wraps a ``dspy.Module`` as a LangChain ``BaseTool``.

    The wrapped module can then be used inside any LangChain agent or
    LangGraph workflow that expects a list of tools.

    Args:
        module: An instantiated DSPy module.
        name: Tool name.  Defaults to the module's class name (snake_cased).
        description: Tool description for the agent's prompt.  Defaults to
                     the module's class docstring or a generic description.
        input_key: Which key from the module's Signature to use as the tool
                   input when the agent passes a plain string.  Defaults to
                   the first input field of the Signature, or ``"input"`` as
                   fallback.
        output_key: Which key from the module's output to return as the tool
                    result string.  Defaults to the first output field.

    Example:
        >>> import dspy
        >>> from dspy_lc_bridge import DSPyTool
        >>>
        >>> summarizer = dspy.ChainOfThought("document -> summary")
        >>> tool = DSPyTool(summarizer, name="summarize", description="Summarize a document.")
        >>>
        >>> # Use in a LangChain/LangGraph agent
        >>> agent = create_react_agent(llm, tools=[tool])
    """

    def __init__(
        self,
        module: Any,
        name: str | None = None,
        description: str | None = None,
        input_key: str | None = None,
        output_key: str | None = None,
    ) -> None:
        self.module = module
        self.name: str = name or _snake_case(module.__class__.__name__)
        self.description: str = description or self._infer_description(module)
        self._input_key = input_key or self._infer_input_key(module)
        self._output_key = output_key or self._infer_output_key(module)

    # ------------------------------------------------------------------
    # LangChain BaseTool protocol (duck-typed)
    # ------------------------------------------------------------------

    def _run(self, tool_input: str, **kwargs: Any) -> str:
        """Invoke the DSPy module with *tool_input* and return a string result.

        Args:
            tool_input: The raw string input provided by the agent.

        Returns:
            The module's output as a string.
        """
        module_input = self._parse_input(tool_input)
        prediction = self.module(**module_input)
        return self._extract_output(prediction)

    async def _arun(self, tool_input: str, **kwargs: Any) -> str:
        """Async version of ``_run``.

        Args:
            tool_input: The raw string input provided by the agent.

        Returns:
            The module's output as a string.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._run, tool_input)

    def run(self, tool_input: str, **kwargs: Any) -> str:
        """Public synchronous entry point (matches LangChain ``BaseTool.run``)."""
        return self._run(tool_input, **kwargs)

    async def arun(self, tool_input: str, **kwargs: Any) -> str:
        """Public async entry point."""
        return await self._arun(tool_input, **kwargs)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _parse_input(self, tool_input: str) -> dict[str, Any]:
        """Parse a raw agent string into a dict for the DSPy module.

        Tries JSON first; falls back to assigning the entire string to the
        default input key.
        """
        try:
            parsed = json.loads(tool_input)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass
        return {self._input_key: tool_input}

    def _extract_output(self, prediction: Any) -> str:
        """Extract a string result from a DSPy Prediction."""
        if hasattr(prediction, self._output_key):
            return str(getattr(prediction, self._output_key))
        if hasattr(prediction, "items"):
            items = dict(prediction.items())
            if self._output_key in items:
                return str(items[self._output_key])
            # Return first available value
            if items:
                return str(next(iter(items.values())))
        return str(prediction)

    @staticmethod
    def _infer_description(module: Any) -> str:
        doc = inspect.getdoc(module) or inspect.getdoc(module.__class__) or ""
        return doc.strip() or f"A DSPy {module.__class__.__name__} module."

    @staticmethod
    def _infer_input_key(module: Any) -> str:
        sig = getattr(module, "signature", None) or getattr(module, "__signature__", None)
        if sig and hasattr(sig, "input_fields") and sig.input_fields:
            return str(next(iter(sig.input_fields)))
        return "input"

    @staticmethod
    def _infer_output_key(module: Any) -> str:
        sig = getattr(module, "signature", None) or getattr(module, "__signature__", None)
        if sig and hasattr(sig, "output_fields") and sig.output_fields:
            return str(next(iter(sig.output_fields)))
        return "output"

    def __repr__(self) -> str:
        return f"DSPyTool(name={self.name!r}, module={self.module.__class__.__name__})"


# ---------------------------------------------------------------------------
# Bridge 8: LangChain Tool → DSPy-compatible tool
# ---------------------------------------------------------------------------


class LangChainTool:
    """Wraps a LangChain ``BaseTool`` for use inside DSPy agent loops.

    DSPy's ``dspy.ReAct`` and custom agent modules expect tools that are
    plain callables with ``name`` and ``desc`` attributes.  This adapter
    satisfies that contract.

    Args:
        tool: A LangChain ``BaseTool`` instance.

    Example:
        >>> from langchain_community.tools import WikipediaQueryRun
        >>> from dspy_lc_bridge import LangChainTool
        >>>
        >>> wiki = LangChainTool(WikipediaQueryRun())
        >>> react = dspy.ReAct("question -> answer", tools=[wiki])
    """

    def __init__(self, tool: Any) -> None:
        self._tool = tool
        self.name: str = getattr(tool, "name", tool.__class__.__name__)
        self.desc: str = getattr(tool, "description", "")

    def __call__(self, input: str, **kwargs: Any) -> str:
        """Invoke the LangChain tool.

        Args:
            input: The tool input string.

        Returns:
            The tool's output as a string.
        """
        if hasattr(self._tool, "run"):
            return str(self._tool.run(input, **kwargs))
        if hasattr(self._tool, "invoke"):
            result = self._tool.invoke(input, **kwargs)
            return str(result)
        raise AttributeError(f"Tool {self._tool!r} has neither 'run' nor 'invoke'.")

    async def acall(self, input: str, **kwargs: Any) -> str:
        """Async invocation of the LangChain tool.

        Args:
            input: The tool input string.

        Returns:
            The tool's output as a string.
        """
        if hasattr(self._tool, "arun"):
            return str(await self._tool.arun(input, **kwargs))
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.__call__, input)

    def __repr__(self) -> str:
        return f"LangChainTool(name={self.name!r})"


# ---------------------------------------------------------------------------
# Bridge 9: DSPy Module → LangGraph node function
# ---------------------------------------------------------------------------


def DSPyNode(
    module: Any,
    input_map: dict[str, str] | None = None,
    output_map: dict[str, str] | None = None,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Create a LangGraph-compatible node function from a DSPy module.

    A LangGraph node function receives the full graph state dict and returns
    a partial update dict.  This factory function creates such a callable by
    mapping state keys to DSPy Signature fields and back.

    Args:
        module: An instantiated DSPy module.
        input_map: Mapping from ``{state_key: signature_input_field}``.
                   Defaults to identity (state keys == signature input fields).
        output_map: Mapping from ``{signature_output_field: state_key}``.
                   Defaults to identity (output fields written directly to state).

    Returns:
        A function ``(state: dict) -> dict`` suitable for use as a LangGraph node.

    Example:
        >>> import dspy
        >>> from langgraph.graph import StateGraph
        >>> from dspy_lc_bridge import DSPyNode
        >>>
        >>> cot = dspy.ChainOfThought("question -> answer")
        >>> node_fn = DSPyNode(cot)
        >>>
        >>> graph = StateGraph(dict)
        >>> graph.add_node("reason", node_fn)
    """

    def node_fn(state: dict[str, Any]) -> dict[str, Any]:
        # Build DSPy input from state
        if input_map:
            module_input = {
                sig_field: state[state_key]
                for state_key, sig_field in input_map.items()
                if state_key in state
            }
        else:
            # Pass all state keys as-is; DSPy will use what it needs
            module_input = dict(state)

        prediction = module(**module_input)

        # Extract outputs
        if hasattr(prediction, "_store"):
            raw = dict(prediction._store)
        elif hasattr(prediction, "items"):
            raw = dict(prediction.items())
        else:
            raw = {"output": str(prediction)}

        if output_map:
            return {
                state_key: raw[out_field]
                for out_field, state_key in output_map.items()
                if out_field in raw
            }
        return raw

    # Attach metadata for introspection
    node_fn.__name__ = _snake_case(module.__class__.__name__)
    node_fn.__doc__ = f"LangGraph node wrapping dspy.{module.__class__.__name__}"
    return node_fn


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _snake_case(name: str) -> str:
    """Convert CamelCase to snake_case."""
    import re

    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()
