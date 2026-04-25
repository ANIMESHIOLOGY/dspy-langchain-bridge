"""DSPy Signature ↔ LangChain PromptTemplate conversion utilities.

Provides two-way conversion between DSPy's structured ``Signature`` abstraction
and LangChain's ``PromptTemplate`` / ``ChatPromptTemplate``, enabling reuse of
prompt definitions across both frameworks.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# DSPy Signature → LangChain PromptTemplate
# ---------------------------------------------------------------------------


def _get_signature_fields(sig: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (input_fields, output_fields) from a DSPy Signature.

    Handles both the class-based DSPy 2.x Signature (accessed via the class
    itself) and instance-level access patterns.
    """
    # DSPy 2.x: Signature is a class; fields live on the class
    if hasattr(sig, "input_fields") and hasattr(sig, "output_fields"):
        return dict(sig.input_fields), dict(sig.output_fields)
    raise TypeError(
        f"Cannot extract fields from {type(sig)!r}. "
        "Expected a dspy.Signature class or instance."
    )


def signature_to_prompt(sig: Any, include_output_hint: bool = False) -> Any:
    """Convert a DSPy Signature to a LangChain PromptTemplate.

    Each input field in the Signature becomes a template variable
    (``{field_name}``).  Field descriptions, when present, are added as
    comments above the corresponding placeholder.

    Args:
        sig: A DSPy ``Signature`` class or instance with ``input_fields``
             and ``output_fields`` attributes.
        include_output_hint: If ``True``, append a hint listing the expected
                             output fields at the end of the template.

    Returns:
        A LangChain ``PromptTemplate`` whose ``input_variables`` match the
        Signature's input field names.

    Example:
        >>> import dspy
        >>> from dspy_lc_bridge import signature_to_prompt
        >>> sig = dspy.Signature("question -> answer")
        >>> prompt = signature_to_prompt(sig)
        >>> prompt.input_variables
        ['question']
        >>> prompt.format(question="What is 2+2?")
        'question: {question}'
    """
    from langchain_core.prompts import PromptTemplate

    input_fields, output_fields = _get_signature_fields(sig)

    parts: list[str] = []
    for field_name, field_info in input_fields.items():
        desc = _get_field_desc(field_info)
        if desc:
            parts.append(f"# {desc}")
        parts.append(f"{field_name}: {{{field_name}}}")

    if include_output_hint and output_fields:
        out_names = ", ".join(output_fields.keys())
        parts.append(f"\n# Expected outputs: {out_names}")

    template = "\n".join(parts)
    return PromptTemplate(
        input_variables=list(input_fields.keys()),
        template=template,
    )


def signature_to_chat_prompt(sig: Any) -> Any:
    """Convert a DSPy Signature to a LangChain ChatPromptTemplate.

    The system message describes the task (derived from the Signature's
    ``__doc__`` or output fields), and a human message contains the input
    variables.

    Args:
        sig: A DSPy ``Signature`` class or instance.

    Returns:
        A LangChain ``ChatPromptTemplate``.

    Example:
        >>> import dspy
        >>> from dspy_lc_bridge import signature_to_chat_prompt
        >>> sig = dspy.Signature("question -> answer")
        >>> chat_prompt = signature_to_chat_prompt(sig)
    """
    from langchain_core.prompts import ChatPromptTemplate

    input_fields, output_fields = _get_signature_fields(sig)

    # Build system message from docstring or output field names
    doc = getattr(sig, "__doc__", None) or ""
    if not doc.strip():
        out_names = ", ".join(output_fields.keys())
        doc = f"Given the inputs, produce: {out_names}."

    # Build human message template
    human_parts: list[str] = []
    for field_name, field_info in input_fields.items():
        desc = _get_field_desc(field_info)
        label = desc if desc else field_name
        human_parts.append(f"{label}: {{{field_name}}}")
    human_template = "\n".join(human_parts)

    return ChatPromptTemplate.from_messages(
        [
            ("system", doc.strip()),
            ("human", human_template),
        ]
    )


# ---------------------------------------------------------------------------
# LangChain PromptTemplate → DSPy Signature
# ---------------------------------------------------------------------------


def prompt_to_signature(
    prompt: Any,
    output_fields: list[str] | None = None,
) -> Any:
    """Convert a LangChain PromptTemplate to a DSPy Signature.

    The Signature's input fields are taken from the prompt's
    ``input_variables``.  Since PromptTemplates have no notion of output
    fields, you must supply them explicitly (or accept the default ``answer``).

    Args:
        prompt: A LangChain ``PromptTemplate`` or ``ChatPromptTemplate``.
        output_fields: List of output field names for the Signature.
                       Defaults to ``["answer"]`` if not provided.

    Returns:
        A ``dspy.Signature`` class whose input fields match the prompt
        variables and output fields match ``output_fields``.

    Example:
        >>> from langchain_core.prompts import PromptTemplate
        >>> from dspy_lc_bridge import prompt_to_signature
        >>> prompt = PromptTemplate.from_template("Answer: {question}")
        >>> sig = prompt_to_signature(prompt, output_fields=["answer"])
        >>> sig.input_fields.keys()
        dict_keys(['question'])
    """
    import dspy

    if output_fields is None:
        output_fields = ["answer"]

    # Collect input variable names
    if hasattr(prompt, "input_variables"):
        input_vars: list[str] = list(prompt.input_variables)
    elif hasattr(prompt, "messages"):
        # ChatPromptTemplate — gather variables from all message templates
        input_vars = list(prompt.input_variables)
    else:
        raise TypeError(
            f"Cannot extract input_variables from {type(prompt)!r}."
        )

    sig_str = ", ".join(input_vars) + " -> " + ", ".join(output_fields)
    return dspy.Signature(sig_str)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_field_desc(field_info: Any) -> str:
    """Extract a human-readable description from a DSPy field definition."""
    # DSPy 2.x stores desc in json_schema_extra
    if hasattr(field_info, "json_schema_extra"):
        extra = field_info.json_schema_extra or {}
        return str(extra.get("desc", "")).strip()
    # Older DSPy or plain dict
    if isinstance(field_info, dict):
        return str(field_info.get("desc", "")).strip()
    return ""
