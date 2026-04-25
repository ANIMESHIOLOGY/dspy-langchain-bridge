# RAG Pipeline

## LangChain Retriever → DSPy

Use any LangChain retriever (FAISS, Chroma, Pinecone) inside a DSPy RAG module:

```python
import dspy
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dspy_lc_bridge import LangChainRetriever, LangChainLM

dspy.settings.configure(lm=LangChainLM(ChatOpenAI(model="gpt-4o")))

vectorstore = FAISS.load_local("my_index", OpenAIEmbeddings())
retriever = LangChainRetriever(vectorstore.as_retriever(), k=3)

class RAG(dspy.Module):
    def __init__(self):
        self.retrieve = retriever
        self.generate = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        passages = self.retrieve(question).passages
        return self.generate(context="\n".join(passages), question=question)

rag = RAG()
result = rag(question="What is the capital of France?")
print(result.answer)
```

## DSPy Retrieve → LangChain

Expose a DSPy `Retrieve` module as a standard LangChain `BaseRetriever`:

```python
import dspy
from dspy_lc_bridge import DSPyRetriever

dspy_rm = dspy.Retrieve(k=5)
retriever = DSPyRetriever(dspy_rm)

# Now use it anywhere LangChain expects a retriever
docs = retriever.invoke("France capital")
```
