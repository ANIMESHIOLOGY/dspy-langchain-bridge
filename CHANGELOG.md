# Changelog

All notable changes to `dspy-langchain-bridge` will be documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

## [0.4.0] — 2026-04-26

### Added — Phase 4: Optimizer Bridge
- `optimize_langchain_prompt` — run DSPy `BootstrapFewShot`, `MIPROv2`, or `COPRO`
  on any LangChain `PromptTemplate` / `ChatPromptTemplate` and get back an optimized
  template with few-shot examples and improved instructions injected

## [0.3.0] — 2026-04-26

### Added — Phase 3: Agent & Tool Interop
- `DSPyTool` — wrap any `dspy.Module` as a LangChain `BaseTool` for use in agents
- `LangChainTool` — use any LangChain tool inside DSPy agent loops (`dspy.ReAct`)
- `DSPyNode` — factory that converts a `dspy.Module` into a LangGraph node function
- `DSPyCallbackHandler` — LangSmith tracing passthrough for DSPy calls inside LangChain
- `instrument_module` — patch a DSPy module for automatic trace recording

## [0.2.0] — 2026-04-26

### Added — Phase 2: Retriever & Memory Bridge
- `LangChainRetriever` — wrap any LangChain `BaseRetriever` (FAISS, Chroma, Pinecone …)
  as a DSPy `Retrieve`-compatible module; supports async via `acall`
- `DSPyRetriever` — expose a DSPy `Retrieve` module as a LangChain `BaseRetriever`;
  supports `invoke`, `ainvoke`, and legacy `get_relevant_documents`
- `LangChainMemoryAdapter` — bridge LangChain conversation memory
  (`ConversationBufferMemory`, `ConversationSummaryMemory`, …) into DSPy history format

## [0.1.0] — 2026-04-26

### Added — Phase 1: Core Bridge Layer
- `DSPyRunnable` — wrap any `dspy.Module` as a LangChain `Runnable` for LCEL chains;
  supports `invoke`, `ainvoke`, `batch`, `abatch`
- `LangChainLM` — use any LangChain `BaseLLM` / `BaseChatModel` as a DSPy LM backend;
  auto-detects chat vs completion model; tracks call history
- `signature_to_prompt` — convert a `dspy.Signature` to a LangChain `PromptTemplate`
- `signature_to_chat_prompt` — convert a `dspy.Signature` to a `ChatPromptTemplate`
- `prompt_to_signature` — convert a LangChain `PromptTemplate` to a `dspy.Signature`
- Project scaffold: `pyproject.toml`, MIT license, CI/CD workflows, mkdocs docs site,
  94-test unit suite
