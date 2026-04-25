# Compatibility Matrix

This document maps DSPy primitives to their LangChain equivalents and shows the bridge status for each.

| DSPy Primitive | LangChain Equivalent | Bridge | Status |
|---|---|---|---|
| `dspy.Module` | `Runnable` (LCEL) | `DSPyRunnable` | ✅ Implemented |
| Any `BaseLLM` / `BaseChatModel` | DSPy LM backend | `LangChainLM` | ✅ Implemented |
| `dspy.Signature` | `PromptTemplate` | `signature_to_prompt` | ✅ Implemented |
| `dspy.Signature` | `ChatPromptTemplate` | `signature_to_chat_prompt` | ✅ Implemented |
| `PromptTemplate` | `dspy.Signature` | `prompt_to_signature` | ✅ Implemented |
| `BaseRetriever` | `dspy.Retrieve` | `LangChainRetriever` | ✅ Implemented |
| `dspy.Retrieve` | `BaseRetriever` | `DSPyRetriever` | ✅ Implemented |
| `dspy.Module` | `BaseTool` | `DSPyTool` | ✅ Implemented |
| `BaseTool` | DSPy tool | `LangChainTool` | ✅ Implemented |
| `dspy.Module` | LangGraph node fn | `DSPyNode` | ✅ Implemented |
| `BaseMemory` | DSPy history | `LangChainMemoryAdapter` | ✅ Implemented |
| DSPy optimizer | LangChain prompt optimizer | `optimize_langchain_prompt` | ✅ Implemented |
| LangSmith tracing | DSPy call spans | `DSPyCallbackHandler` | ✅ Implemented |

## Async Support

| Bridge | `invoke` | `ainvoke` | Streaming |
|---|---|---|---|
| `DSPyRunnable` | ✅ | ✅ | Planned v1.0 |
| `LangChainRetriever` | ✅ | ✅ | N/A |
| `DSPyRetriever` | ✅ | ✅ | N/A |
| `DSPyTool` | ✅ | ✅ | N/A |
| `LangChainTool` | ✅ | ✅ | N/A |

## DSPy Version Support

| DSPy Version | Supported |
|---|---|
| `dspy-ai >= 2.0.0` | ✅ |
| `dspy-ai < 2.0.0` (v1.x) | ❌ Not supported |

## LangChain Version Support

| Package | Minimum Version |
|---|---|
| `langchain` | `>= 0.2.0` |
| `langchain-core` | `>= 0.2.0` |
| `langgraph` | `>= 0.1.0` (optional) |
