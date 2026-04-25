"""Shared internal type definitions for dspy-langchain-bridge."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

# DSPy Prediction fields as plain Python dicts after serialization
PredictionDict = Dict[str, Any]

# Message in OpenAI-style chat format
ChatMessage = Dict[str, str]  # {"role": "...", "content": "..."}

# A trainset example — dict with input/output keys matching a DSPy Signature
TrainExample = Dict[str, Any]

# Metric function signature: (example, prediction, trace=None) -> float | bool
MetricFn = Any  # Callable[[Any, Any, Optional[Any]], Union[float, bool]]

# DSPy optimizer type union (for type hints without importing DSPy at module level)
DSPyOptimizer = Any

# LangChain LLM / ChatModel union
LangChainLLMType = Any

# LangChain PromptTemplate or ChatPromptTemplate union
LangChainPromptType = Any

__all__: List[str] = [
    "PredictionDict",
    "ChatMessage",
    "TrainExample",
    "MetricFn",
    "DSPyOptimizer",
    "LangChainLLMType",
    "LangChainPromptType",
]
