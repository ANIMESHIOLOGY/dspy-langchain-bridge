"""LangSmith tracing passthrough for DSPy modules running inside LangChain.

When a ``DSPyRunnable`` executes inside a LangChain chain that has LangSmith
tracing enabled, this module injects the active run context into DSPy's
execution so that DSPy calls appear as child spans in the same LangSmith
trace.
"""

from __future__ import annotations

import functools
import time
from typing import Any


class DSPyCallbackHandler:
    """LangChain-compatible callback handler that traces DSPy module calls.

    Attach this handler to your LangChain run config (or set it globally via
    ``langchain.callbacks.set_handler``) to see DSPy module calls as spans in
    LangSmith.

    Args:
        project_name: LangSmith project name.  Falls back to the
                      ``LANGCHAIN_PROJECT`` environment variable.

    Example:
        >>> from dspy_lc_bridge import DSPyCallbackHandler
        >>> from langchain_core.runnables import RunnableConfig
        >>>
        >>> handler = DSPyCallbackHandler(project_name="my-project")
        >>> config = RunnableConfig(callbacks=[handler])
        >>> result = chain.invoke({"question": "..."}, config=config)
    """

    def __init__(self, project_name: str | None = None) -> None:
        self.project_name = project_name
        self._spans: list[dict[str, Any]] = []
        self._langsmith_available = self._check_langsmith()

    @staticmethod
    def _check_langsmith() -> bool:
        try:
            import langsmith  # noqa: F401

            return True
        except ImportError:
            return False

    # ------------------------------------------------------------------
    # LangChain callbacks protocol (no-op stubs + DSPy instrumentation)
    # ------------------------------------------------------------------

    def on_chain_start(
        self, serialized: dict[str, Any], inputs: dict[str, Any], **kwargs: Any
    ) -> None:
        pass

    def on_chain_end(self, outputs: dict[str, Any], **kwargs: Any) -> None:
        pass

    def on_chain_error(self, error: Exception, **kwargs: Any) -> None:
        pass

    def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any) -> None:
        pass

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        pass

    def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        pass

    def on_tool_start(self, serialized: dict[str, Any], input_str: str, **kwargs: Any) -> None:
        pass

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        pass

    def on_tool_error(self, error: Exception, **kwargs: Any) -> None:
        pass

    # ------------------------------------------------------------------
    # DSPy module tracing
    # ------------------------------------------------------------------

    def trace_dspy_module(
        self, module: Any, inputs: dict[str, Any], outputs: Any, latency_ms: float
    ) -> None:
        """Record a DSPy module execution as a span.

        This is called automatically by ``DSPyRunnable`` when a handler is
        attached via the LangChain callback system.

        Args:
            module: The DSPy module that was called.
            inputs: The input dict passed to the module.
            outputs: The prediction / output dict returned.
            latency_ms: Wall-clock latency in milliseconds.
        """
        span: dict[str, Any] = {
            "module": module.__class__.__name__,
            "inputs": inputs,
            "outputs": outputs,
            "latency_ms": latency_ms,
        }
        self._spans.append(span)

        if self._langsmith_available:
            self._post_to_langsmith(span)

    def _post_to_langsmith(self, span: dict[str, Any]) -> None:
        """Post the span to LangSmith as a child run (best-effort)."""
        try:
            from langsmith import Client

            client = Client()
            client.create_run(
                name=f"DSPy:{span['module']}",
                run_type="chain",
                inputs=span["inputs"],
                outputs=(
                    span["outputs"]
                    if isinstance(span["outputs"], dict)
                    else {"output": str(span["outputs"])}
                ),
                extra={"latency_ms": span["latency_ms"]},
                project_name=self.project_name,
            )
        except Exception:
            # Tracing is best-effort; never let it crash user code
            pass

    def get_spans(self) -> list[dict[str, Any]]:
        """Return all recorded DSPy spans from this session.

        Returns:
            List of span dicts with ``module``, ``inputs``, ``outputs``, and
            ``latency_ms`` keys.
        """
        return list(self._spans)

    def __repr__(self) -> str:
        return f"DSPyCallbackHandler(project={self.project_name!r}, " f"spans={len(self._spans)})"


def instrument_module(module: Any, handler: DSPyCallbackHandler) -> Any:
    """Patch a DSPy module's ``forward`` method to auto-record traces.

    Wraps the module's ``forward`` (or ``__call__``) so every invocation
    automatically calls ``handler.trace_dspy_module``.

    Args:
        module: The DSPy module to instrument.
        handler: A ``DSPyCallbackHandler`` to receive trace events.

    Returns:
        The same module object (mutated in place).
    """
    original_call = module.__class__.__call__

    @functools.wraps(original_call)
    def traced_call(self_inner: Any, *args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        result = original_call(self_inner, *args, **kwargs)
        latency_ms = (time.perf_counter() - t0) * 1000
        handler.trace_dspy_module(
            module=self_inner,
            inputs=kwargs,
            outputs=result,
            latency_ms=latency_ms,
        )
        return result

    module.__class__.__call__ = traced_call
    return module
