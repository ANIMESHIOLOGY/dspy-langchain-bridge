# Basic Runnable — DSPy Module in an LCEL Chain

## Overview

`DSPyRunnable` wraps any `dspy.Module` so it can be used inside LangChain's
LCEL pipe syntax.

## Example

```python
import dspy
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dspy_lc_bridge import DSPyRunnable, LangChainLM

# 1. Configure DSPy to use a LangChain LLM
llm = ChatOpenAI(model="gpt-4o")
dspy.settings.configure(lm=LangChainLM(llm))

# 2. Build a DSPy module
cot = dspy.ChainOfThought("question -> answer")

# 3. Build an LCEL chain
prompt = ChatPromptTemplate.from_template("Question: {question}")
chain = prompt | DSPyRunnable(cot)

# 4. Invoke
result = chain.invoke({"question": "What is the capital of France?"})
print(result["answer"])  # → "Paris"
```

## Notes

- The `DSPyRunnable.invoke` returns a `dict` matching the Signature's output fields.
- `ainvoke` is supported for async chains.
- DSPy modules are synchronous; `ainvoke` uses a thread-pool executor.
