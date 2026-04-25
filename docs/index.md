# dspy-langchain-bridge

A production-grade compatibility bridge between [DSPy](https://github.com/stanfordnlp/dspy) and [LangChain](https://github.com/langchain-ai/langchain).

## Quick Start

```bash
pip install dspy-langchain-bridge
```

```python
import dspy
from langchain_openai import ChatOpenAI
from dspy_lc_bridge import DSPyRunnable, LangChainLM

llm = ChatOpenAI(model="gpt-4o")
dspy.settings.configure(lm=LangChainLM(llm))

cot = dspy.ChainOfThought("question -> answer")
chain = some_prompt | DSPyRunnable(cot)
result = chain.invoke({"question": "What is the capital of France?"})
```

## Navigation

- [Installation](installation.md)
- [Compatibility Matrix](compatibility-matrix.md)
- [Getting Started](getting-started/basic-runnable.md)
- [API Reference](api-reference/runnables.md)
- [Migration Guide](migration-guide.md)
