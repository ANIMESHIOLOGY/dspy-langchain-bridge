# Migration Guide

## Moving from Pure LangChain to LangChain + DSPy

### Step 1: Add DSPy optimization to an existing LangChain chain

**Before** (pure LangChain):

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

chain = ChatPromptTemplate.from_template("{question}") | ChatOpenAI(model="gpt-4o")
result = chain.invoke({"question": "What is 2+2?"})
```

**After** (with DSPy optimization):

```python
import dspy
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dspy.teleprompt import BootstrapFewShot
from dspy_lc_bridge import LangChainLM, optimize_langchain_prompt

dspy.settings.configure(lm=LangChainLM(ChatOpenAI(model="gpt-4o")))

prompt = ChatPromptTemplate.from_template("{question}")
optimized_prompt = optimize_langchain_prompt(prompt, BootstrapFewShot(metric=my_metric), trainset)

chain = optimized_prompt | ChatOpenAI(model="gpt-4o")
```

### Step 2: Replace a LangChain retriever with a DSPy-optimized RAG module

**Before**:

```python
from langchain.chains import RetrievalQA
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
```

**After** (DSPy RAG inside LangChain):

```python
from dspy_lc_bridge import DSPyRunnable, LangChainRetriever

lc_retriever = LangChainRetriever(retriever, k=3)

class RAG(dspy.Module):
    ...

chain = some_prompt | DSPyRunnable(RAG())
```

### Step 3: Add DSPy ChainOfThought to a LangGraph agent

```python
from langgraph.graph import StateGraph
from dspy_lc_bridge import DSPyNode

cot = dspy.ChainOfThought("question -> answer")
graph.add_node("reason", DSPyNode(cot))
```
