# Changelog

All notable changes to `dspy-langchain-bridge` will be documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

## [0.1.0] — 2026-04-26

### Added
- `DSPyRunnable` — wrap any `dspy.Module` as a LangChain `Runnable` for use in LCEL chains
- `LangChainLM` — use any LangChain `BaseLLM` / `BaseChatModel` as a DSPy LM backend
- `signature_to_prompt` — convert a `dspy.Signature` to a LangChain `PromptTemplate`
- `prompt_to_signature` — convert a LangChain `PromptTemplate` to a `dspy.Signature`
- `LangChainRetriever` — wrap a LangChain `BaseRetriever` as a DSPy `Retrieve` module
- `DSPyRetriever` — expose a DSPy `Retrieve` module as a LangChain `BaseRetriever`
- `LangChainMemoryAdapter` — bridge LangChain conversation memory into DSPy history format
- `DSPyTool` — wrap a DSPy module as a LangChain `BaseTool`
- `LangChainTool` — use any LangChain tool inside a DSPy agent loop
- `DSPyNode` — wrap a DSPy module as a LangGraph node function
- `optimize_langchain_prompt` — run DSPy optimizers on LangChain `PromptTemplate` objects
- Full async support (`ainvoke`, `_aget_relevant_documents`, `_arun`) across all bridges
- LangSmith tracing passthrough via `DSPyCallbackHandler`
