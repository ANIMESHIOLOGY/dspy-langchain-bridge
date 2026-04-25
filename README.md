# dspy-langchain-bridge

> A production-grade compatibility bridge between [DSPy](https://github.com/stanfordnlp/dspy) and [LangChain](https://github.com/langchain-ai/langchain).

[![PyPI version](https://badge.fury.io/py/dspy-langchain-bridge.svg)](https://pypi.org/project/dspy-langchain-bridge/)
[![CI](https://github.com/ANIMESHIOLOGY/dspy-langchain-bridge/actions/workflows/ci.yml/badge.svg)](https://github.com/ANIMESHIOLOGY/dspy-langchain-bridge/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![DSPy](https://img.shields.io/badge/DSPy-2.x%2F3.x-orange.svg)](https://github.com/stanfordnlp/dspy)
[![LangChain](https://img.shields.io/badge/LangChain-0.2%2B-green.svg)](https://github.com/langchain-ai/langchain)

---

## The Problem

DSPy and LangChain are the two most widely used LLM frameworks — and they don't talk to each other.

- **LangChain** has a massive ecosystem: tools, retrievers, agents, memory, LangGraph.
- **DSPy** has the best prompt optimizers in the ecosystem (`BootstrapFewShot`, `MIPROv2`, `COPRO`).

If you want DSPy's optimization *inside* a LangChain pipeline — or LangChain's retrievers *inside* a DSPy RAG module — you're currently on your own. `dspy-langchain-bridge` solves that.

---

## Installation

```bash
pip install dspy-langchain-bridge
```

With optional extras:

```bash
pip install "dspy-langchain-bridge[openai,langgraph]"   # OpenAI + LangGraph support
pip install "dspy-langchain-bridge[all]"                # Everything
```

---

## Quick Start

### Use a DSPy module inside a LangChain chain

```python
import dspy
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dspy_lc_bridge import DSPyRunnable, LangChainLM

# Configure DSPy to use any LangChain LLM
llm = ChatOpenAI(model="gpt-4o")
dspy.settings.configure(lm=LangChainLM(llm))

# Build a DSPy module
cot = dspy.ChainOfThought("question -> answer")

# Drop it into an LCEL chain
prompt = ChatPromptTemplate.from_template("{question}")
chain = prompt | DSPyRunnable(cot)

result = chain.invoke({"question": "What is the capital of France?"})
print(result["answer"])
```

### Use a LangChain retriever inside DSPy RAG

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dspy_lc_bridge import LangChainRetriever

vectorstore = FAISS.load_local("my_index", OpenAIEmbeddings())
retriever = LangChainRetriever(vectorstore.as_retriever(), k=3)

class RAG(dspy.Module):
    def __init__(self):
        self.retrieve = retriever
        self.generate = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        passages = self.retrieve(question).passages
        context = "\n".join(passages)
        return self.generate(context=context, question=question)
```

### Optimize a LangChain prompt with DSPy

```python
from langchain_core.prompts import ChatPromptTemplate
from dspy.teleprompt import BootstrapFewShot
from dspy_lc_bridge import optimize_langchain_prompt

prompt = ChatPromptTemplate.from_template("Answer the question: {question}")

optimized = optimize_langchain_prompt(
    prompt=prompt,
    optimizer=BootstrapFewShot(metric=my_metric),
    trainset=my_examples,
)
# optimized is a ChatPromptTemplate with few-shot examples injected
```

---

## Compatibility Matrix

| DSPy Primitive | LangChain Equivalent | Bridge Class / Function |
|---|---|---|
| `dspy.Module` | `Runnable` | `DSPyRunnable` |
| LangChain `BaseLLM` / `BaseChatModel` | DSPy LM backend | `LangChainLM` |
| `dspy.Signature` | `PromptTemplate` | `signature_to_prompt` / `prompt_to_signature` |
| LangChain `BaseRetriever` | `dspy.Retrieve` | `LangChainRetriever` |
| `dspy.Retrieve` | LangChain `BaseRetriever` | `DSPyRetriever` |
| `dspy.Module` | LangChain `BaseTool` | `DSPyTool` |
| LangChain `BaseTool` | DSPy tool | `LangChainTool` |
| `dspy.Module` | LangGraph node function | `DSPyNode` |
| LangChain `BaseMemory` | DSPy history | `LangChainMemoryAdapter` |
| DSPy optimizer | LangChain prompt optimizer | `optimize_langchain_prompt` |

---

## Public API

```python
from dspy_lc_bridge import (
    # Core
    DSPyRunnable,              # DSPy module → LangChain Runnable
    LangChainLM,               # LangChain LLM → DSPy LM backend

    # Signatures
    signature_to_prompt,       # dspy.Signature → PromptTemplate
    prompt_to_signature,       # PromptTemplate → dspy.Signature

    # Retrievers
    LangChainRetriever,        # LangChain retriever → DSPy Retrieve
    DSPyRetriever,             # DSPy Retrieve → LangChain BaseRetriever

    # Tools & Agents
    DSPyTool,                  # DSPy module → LangChain Tool
    LangChainTool,             # LangChain Tool → DSPy tool
    DSPyNode,                  # DSPy module → LangGraph node

    # Memory
    LangChainMemoryAdapter,    # LangChain memory → DSPy history

    # Optimizers
    optimize_langchain_prompt, # DSPy optimizer → optimized LangChain prompt
)
```

---

## Roadmap

| Version | Contents | Status |
|---------|----------|--------|
| `v0.1.0` | Core bridge: `DSPyRunnable`, `LangChainLM`, signature converters | ✅ Released |
| `v0.2.0` | Retriever & memory bridges | ✅ Released |
| `v0.3.0` | Agent & tool interop + LangGraph bridges | ✅ Released |
| `v0.4.0` | Optimizer bridge | ✅ Released |
| `v1.0.0` | Stable release, full docs, streaming, mypy compliance | 🔜 Planned |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). PRs, issues, and discussions are welcome.

## License

[MIT](LICENSE)
