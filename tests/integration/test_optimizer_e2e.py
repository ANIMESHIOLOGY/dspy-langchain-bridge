"""End-to-end integration tests for the optimizer bridge.

These tests hit real LLM APIs and run DSPy optimizers.
Run with: pytest tests/integration/ -m slow
"""

from __future__ import annotations

import pytest


@pytest.mark.slow
@pytest.mark.integration
def test_bootstrap_fewshot_optimizes_langchain_prompt() -> None:
    """Run BootstrapFewShot on a LangChain PromptTemplate."""
    import dspy
    from dspy.teleprompt import BootstrapFewShot
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI

    from dspy_lc_bridge import LangChainLM, optimize_langchain_prompt

    llm = ChatOpenAI(model="gpt-4o-mini")
    dspy.settings.configure(lm=LangChainLM(llm))

    prompt = ChatPromptTemplate.from_template("Answer the question: {question}")

    trainset = [
        dspy.Example(question="What is 2+2?", answer="4").with_inputs("question"),
        dspy.Example(question="What is 3+3?", answer="6").with_inputs("question"),
        dspy.Example(question="What is the capital of France?", answer="Paris").with_inputs("question"),
    ]

    def exact_match(example: dspy.Example, prediction: dspy.Prediction, trace: None = None) -> bool:
        return str(prediction.answer).strip().lower() == str(example.answer).strip().lower()

    optimizer = BootstrapFewShot(metric=exact_match, max_bootstrapped_demos=2)

    optimized = optimize_langchain_prompt(
        prompt=prompt,
        optimizer=optimizer,
        trainset=trainset,
        output_fields=["answer"],
    )

    from langchain_core.prompts import ChatPromptTemplate

    assert isinstance(optimized, ChatPromptTemplate)
    # Optimized prompt should have more messages than the original (demos injected)
    assert len(optimized.messages) >= 2
