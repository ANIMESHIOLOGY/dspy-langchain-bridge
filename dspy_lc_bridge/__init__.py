"""dspy-langchain-bridge — compatibility layer between DSPy and LangChain.

Public API
----------
Core
~~~~
DSPyRunnable            — dspy.Module → LangChain Runnable (LCEL)
LangChainLM             — LangChain LLM → DSPy LM backend

Signatures
~~~~~~~~~~
signature_to_prompt     — dspy.Signature → PromptTemplate
signature_to_chat_prompt— dspy.Signature → ChatPromptTemplate
prompt_to_signature     — PromptTemplate → dspy.Signature

Retrievers
~~~~~~~~~~
LangChainRetriever      — LangChain BaseRetriever → DSPy Retrieve
DSPyRetriever           — DSPy Retrieve → LangChain BaseRetriever

Tools & Agents
~~~~~~~~~~~~~~
DSPyTool                — dspy.Module → LangChain BaseTool
LangChainTool           — LangChain Tool → DSPy tool
DSPyNode                — dspy.Module → LangGraph node function

Memory
~~~~~~
LangChainMemoryAdapter  — LangChain Memory → DSPy history

Optimizers
~~~~~~~~~~
optimize_langchain_prompt — DSPy optimizer → optimized LangChain prompt

Callbacks / Tracing
~~~~~~~~~~~~~~~~~~~
DSPyCallbackHandler     — LangSmith tracing passthrough
instrument_module       — Patch a DSPy module for auto-tracing
"""

from dspy_lc_bridge.callbacks import DSPyCallbackHandler, instrument_module
from dspy_lc_bridge.llms import LangChainLM
from dspy_lc_bridge.memory import LangChainMemoryAdapter
from dspy_lc_bridge.optimizers import optimize_langchain_prompt
from dspy_lc_bridge.retrievers import DSPyRetriever, LangChainRetriever
from dspy_lc_bridge.runnables import DSPyRunnable
from dspy_lc_bridge.signatures import (
    prompt_to_signature,
    signature_to_chat_prompt,
    signature_to_prompt,
)
from dspy_lc_bridge.tools import DSPyNode, DSPyTool, LangChainTool

__version__ = "0.1.0"

__all__ = [
    # Core
    "DSPyRunnable",
    "LangChainLM",
    # Signatures
    "signature_to_prompt",
    "signature_to_chat_prompt",
    "prompt_to_signature",
    # Retrievers
    "LangChainRetriever",
    "DSPyRetriever",
    # Tools & Agents
    "DSPyTool",
    "LangChainTool",
    "DSPyNode",
    # Memory
    "LangChainMemoryAdapter",
    # Optimizers
    "optimize_langchain_prompt",
    # Callbacks
    "DSPyCallbackHandler",
    "instrument_module",
]
