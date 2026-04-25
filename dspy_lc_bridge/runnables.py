"""DSPy Module → LangChain Runnable bridge.

Wraps any ``dspy.Module`` as a LangChain ``Runnable`` so it can be composed
inside LCEL chains using the pipe (``|``) operator.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Iterator, List, Optional, Type

import dspy
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.runnables.utils import Input, Output

from dspy_lc_bridge._types import PredictionDict


def _prediction_to_dict(prediction: Any) -> PredictionDict:
    """Convert a DSPy Prediction object to a plain dict.

    Handles both dict-like Prediction objects (DSPy 2.x) and plain dicts.
    """
    if isinstance(prediction, dict):
        return dict(prediction)
    # dspy.Prediction exposes items() in 2.x
    if hasattr(prediction, "__dict__") and hasattr(prediction, "_store"):
        return dict(prediction._store)
    if hasattr(prediction, "items"):
        return dict(prediction.items())
    # Fallback: try to convert via vars()
    try:
        return {k: v for k, v in vars(prediction).items() if not k.startswith("_")}
    except TypeError:
        return {"output": str(prediction)}


class DSPyRunnable(Runnable[Dict[str, Any], Dict[str, Any]]):
    """Wraps a ``dspy.Module`` as a LangChain ``Runnable``.

    This lets any compiled or uncompiled DSPy module participate in LCEL
    chains as a first-class component.

    Args:
        module: An instantiated ``dspy.Module`` (e.g. ``dspy.ChainOfThought``).

    Example:
        >>> import dspy
        >>> from langchain_core.prompts import ChatPromptTemplate
        >>> from dspy_lc_bridge import DSPyRunnable
        >>>
        >>> cot = dspy.ChainOfThought("question -> answer")
        >>> chain = ChatPromptTemplate.from_template("{question}") | DSPyRunnable(cot)
        >>> result = chain.invoke({"question": "What is 2+2?"})
        >>> print(result["answer"])
    """

    def __init__(self, module: dspy.Module) -> None:
        self.module = module

    # ------------------------------------------------------------------
    # Runnable protocol
    # ------------------------------------------------------------------

    def invoke(
        self,
        input: Dict[str, Any],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Invoke the DSPy module synchronously.

        Args:
            input: Dict whose keys match the DSPy Signature's input fields.
            config: Optional LangChain run config (callbacks, tags, etc.).

        Returns:
            Dict containing the Signature's output fields plus any extras
            returned by the module.
        """
        # Support LangChain message objects as passthrough (e.g. after a prompt)
        if hasattr(input, "content"):
            # AIMessage / HumanMessage — extract text content
            input = {"text": input.content}  # type: ignore[assignment]

        prediction = self.module(**input)
        return _prediction_to_dict(prediction)

    async def ainvoke(
        self,
        input: Dict[str, Any],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Invoke the DSPy module asynchronously.

        DSPy is synchronous; this runs the call in a thread-pool executor so
        it does not block the event loop.

        Args:
            input: Dict whose keys match the DSPy Signature's input fields.
            config: Optional LangChain run config.

        Returns:
            Dict containing the Signature's output fields.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.invoke, input, config)

    def batch(
        self,
        inputs: List[Dict[str, Any]],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Invoke the DSPy module on a list of inputs.

        Args:
            inputs: List of input dicts.
            config: Optional LangChain run config.

        Returns:
            List of output dicts in the same order as inputs.
        """
        return [self.invoke(inp, config) for inp in inputs]

    async def abatch(
        self,
        inputs: List[Dict[str, Any]],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Async batch invocation.

        Args:
            inputs: List of input dicts.
            config: Optional LangChain run config.

        Returns:
            List of output dicts.
        """
        results = await asyncio.gather(*[self.ainvoke(inp, config) for inp in inputs])
        return list(results)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"DSPyRunnable(module={self.module.__class__.__name__})"

    @property
    def InputType(self) -> Type[Dict[str, Any]]:  # type: ignore[override]
        return Dict[str, Any]  # type: ignore[return-value]

    @property
    def OutputType(self) -> Type[Dict[str, Any]]:  # type: ignore[override]
        return Dict[str, Any]  # type: ignore[return-value]
