# Installation

## Requirements

- Python 3.9+
- `dspy-ai >= 2.0.0`
- `langchain >= 0.2.0`
- `langchain-core >= 0.2.0`

## Basic Install

```bash
pip install dspy-langchain-bridge
```

## With Extras

```bash
# OpenAI LLMs
pip install "dspy-langchain-bridge[openai]"

# LangGraph support
pip install "dspy-langchain-bridge[langgraph]"

# FAISS vector store
pip install "dspy-langchain-bridge[faiss]"

# LangSmith tracing
pip install "dspy-langchain-bridge[langsmith]"

# Everything
pip install "dspy-langchain-bridge[all]"
```

## Development Install

```bash
git clone https://github.com/ANIMESHIOLOGY/dspy-langchain-bridge
cd dspy-langchain-bridge
pip install -e ".[dev]"
pre-commit install
```
