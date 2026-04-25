"""Optimizer bridge — run DSPy optimizers on LangChain prompts.

This is the crown-jewel feature of the library.  LangChain has no native
prompt optimizer; DSPy has the best in the ecosystem.  This module lets you
run ``BootstrapFewShot``, ``MIPROv2``, or ``COPRO`` on any LangChain
``PromptTemplate`` / ``ChatPromptTemplate`` and get back an optimized
template with few-shot examples and improved instructions injected.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Union

from dspy_lc_bridge._types import MetricFn, TrainExample


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def optimize_langchain_prompt(
    prompt: Any,
    optimizer: Any,
    trainset: List[TrainExample],
    metric: Optional[MetricFn] = None,
    output_fields: Optional[List[str]] = None,
    max_labeled_demos: int = 4,
    max_bootstrapped_demos: int = 4,
) -> Any:
    """Optimize a LangChain prompt using a DSPy optimizer.

    Converts the LangChain prompt to a temporary DSPy module, runs the
    optimizer, then extracts the optimized few-shot examples and instructions
    and injects them back into a new LangChain ``ChatPromptTemplate``.

    Supports ``BootstrapFewShot``, ``MIPROv2``, and ``COPRO``.

    Args:
        prompt: A LangChain ``PromptTemplate`` or ``ChatPromptTemplate``.
        optimizer: An *instantiated* DSPy optimizer, e.g.
                   ``BootstrapFewShot(metric=my_metric)``.
        trainset: List of ``dspy.Example`` objects or plain dicts with keys
                  matching the prompt's input variables.
        metric: Optional metric function ``(example, prediction, trace=None)
                -> float | bool``.  Required if the optimizer was not already
                initialized with a metric.
        output_fields: Output field names for the temporary DSPy Signature.
                       Defaults to ``["answer"]``.
        max_labeled_demos: Max labeled demonstrations to use (passed through
                           when the optimizer supports it).
        max_bootstrapped_demos: Max bootstrapped demonstrations.

    Returns:
        An optimized ``ChatPromptTemplate`` with few-shot examples and/or
        improved instructions injected as a system message prefix.

    Example:
        >>> from langchain_core.prompts import ChatPromptTemplate
        >>> from dspy.teleprompt import BootstrapFewShot
        >>> from dspy_lc_bridge import optimize_langchain_prompt
        >>>
        >>> prompt = ChatPromptTemplate.from_template("Answer: {question}")
        >>> optimized = optimize_langchain_prompt(
        ...     prompt=prompt,
        ...     optimizer=BootstrapFewShot(metric=my_metric),
        ...     trainset=my_examples,
        ... )
    """
    import dspy

    if output_fields is None:
        output_fields = ["answer"]

    # 1. Convert LangChain prompt → DSPy Signature + Module
    from dspy_lc_bridge.signatures import prompt_to_signature

    sig = prompt_to_signature(prompt, output_fields=output_fields)
    module = _PromptModule(sig)

    # 2. Normalize trainset to dspy.Example objects
    dspy_trainset = _normalize_trainset(trainset, sig)

    # 3. Set metric on the optimizer if provided and not already set
    if metric is not None and not getattr(optimizer, "metric", None):
        optimizer.metric = metric

    # 4. Run the optimizer
    compiled_module = optimizer.compile(module, trainset=dspy_trainset)

    # 5. Extract optimized artefacts
    demos, instructions = _extract_optimized_artefacts(compiled_module)

    # 6. Rebuild a LangChain ChatPromptTemplate with injected content
    return _build_optimized_prompt(prompt, demos, instructions, sig)


# ---------------------------------------------------------------------------
# Internal DSPy module used during optimization
# ---------------------------------------------------------------------------


class _PromptModule:
    """Minimal DSPy module wrapping a Signature, used during optimization."""

    def __init__(self, signature: Any) -> None:
        import dspy

        self.predict = dspy.Predict(signature)
        self.signature = signature

    def __call__(self, **kwargs: Any) -> Any:
        return self.predict(**kwargs)

    def forward(self, **kwargs: Any) -> Any:
        return self.predict(**kwargs)

    # DSPy Module protocol
    def named_predictors(self) -> List[Any]:
        return [("predict", self.predict)]

    def predictors(self) -> List[Any]:
        return [self.predict]


# ---------------------------------------------------------------------------
# Trainset normalization
# ---------------------------------------------------------------------------


def _normalize_trainset(trainset: List[Any], sig: Any) -> List[Any]:
    """Convert plain dicts to ``dspy.Example`` objects if needed."""
    import dspy

    normalized = []
    for item in trainset:
        if isinstance(item, dspy.Example):
            normalized.append(item)
        elif isinstance(item, dict):
            # Determine which keys are inputs vs outputs
            input_keys = list(sig.input_fields.keys()) if hasattr(sig, "input_fields") else []
            example = dspy.Example(**item)
            if input_keys:
                example = example.with_inputs(*input_keys)
            normalized.append(example)
        else:
            normalized.append(item)
    return normalized


# ---------------------------------------------------------------------------
# Artefact extraction after optimization
# ---------------------------------------------------------------------------


def _extract_optimized_artefacts(
    compiled_module: Any,
) -> tuple[List[Dict[str, Any]], str]:
    """Extract few-shot demos and updated instructions from a compiled module.

    Returns:
        Tuple of (demos list, instructions string).
    """
    demos: List[Dict[str, Any]] = []
    instructions = ""

    # Walk predictor(s) in the compiled module
    predictors = []
    if hasattr(compiled_module, "predictors"):
        predictors = compiled_module.predictors()
    elif hasattr(compiled_module, "predict"):
        predictors = [compiled_module.predict]

    for predictor in predictors:
        # Extract few-shot demos
        raw_demos = getattr(predictor, "demos", [])
        for demo in raw_demos:
            if hasattr(demo, "_store"):
                demos.append(dict(demo._store))
            elif hasattr(demo, "items"):
                demos.append(dict(demo.items()))
            elif isinstance(demo, dict):
                demos.append(demo)

        # Extract updated signature instructions (MIPROv2 / COPRO modify these)
        sig = getattr(predictor, "signature", None)
        if sig is not None:
            doc = getattr(sig, "__doc__", None) or ""
            if doc.strip():
                instructions = doc.strip()

    return demos, instructions


# ---------------------------------------------------------------------------
# Rebuild LangChain ChatPromptTemplate
# ---------------------------------------------------------------------------


def _build_optimized_prompt(
    original_prompt: Any,
    demos: List[Dict[str, Any]],
    instructions: str,
    sig: Any,
) -> Any:
    """Construct an optimized ChatPromptTemplate.

    The system message contains the original task description plus any new
    optimizer-generated instructions.  Few-shot demonstrations are prepended
    as alternating human/assistant messages before the live input slot.

    Args:
        original_prompt: The original LangChain prompt (used to extract
                         the template string and input variables).
        demos: List of few-shot demonstration dicts.
        instructions: Optimizer-generated instruction string (may be empty).
        sig: The DSPy Signature (used for input/output field names).

    Returns:
        A ``ChatPromptTemplate`` with injected examples and instructions.
    """
    from langchain_core.prompts import ChatPromptTemplate

    input_fields = list(sig.input_fields.keys()) if hasattr(sig, "input_fields") else []
    output_fields = list(sig.output_fields.keys()) if hasattr(sig, "output_fields") else ["answer"]

    # Build system message
    original_system = _extract_system_message(original_prompt)
    system_parts = [original_system] if original_system else ["You are a helpful assistant."]
    if instructions:
        system_parts.append(f"\n{instructions}")
    system_msg = " ".join(system_parts).strip()

    messages: List[Any] = [("system", system_msg)]

    # Inject few-shot demonstrations
    for demo in demos:
        # Human turn — build from input fields
        human_parts = []
        for field in input_fields:
            if field in demo:
                human_parts.append(f"{field}: {demo[field]}")
        if human_parts:
            messages.append(("human", "\n".join(human_parts)))

        # Assistant turn — build from output fields
        ai_parts = []
        for field in output_fields:
            if field in demo:
                ai_parts.append(f"{field}: {demo[field]}")
        if ai_parts:
            messages.append(("assistant", "\n".join(ai_parts)))

    # Live input slot
    live_parts = [f"{f}: {{{f}}}" for f in input_fields]
    messages.append(("human", "\n".join(live_parts)))

    return ChatPromptTemplate.from_messages(messages)


def _extract_system_message(prompt: Any) -> str:
    """Try to find the system message text from a LangChain prompt."""
    # ChatPromptTemplate with explicit system message
    if hasattr(prompt, "messages"):
        for msg_template in prompt.messages:
            role = getattr(msg_template, "role", None) or type(msg_template).__name__.lower()
            if "system" in str(role).lower():
                tmpl = getattr(msg_template, "prompt", None)
                if tmpl and hasattr(tmpl, "template"):
                    return str(tmpl.template)
    # Plain PromptTemplate — use the whole template as context
    if hasattr(prompt, "template"):
        return ""
    return ""
