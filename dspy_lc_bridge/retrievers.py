"""Retriever bridges — LangChain ↔ DSPy in both directions.

Bridge 4: ``LangChainRetriever`` — use any LangChain ``BaseRetriever``
          (FAISS, Chroma, Pinecone, …) inside DSPy pipelines.

Bridge 5: ``DSPyRetriever`` — expose a DSPy ``Retrieve`` module as a
          LangChain-compatible ``BaseRetriever``.
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any

# ---------------------------------------------------------------------------
# Bridge 4: LangChain BaseRetriever → DSPy Retrieve
# ---------------------------------------------------------------------------


class LangChainRetriever:
    """Wraps a LangChain ``BaseRetriever`` as a DSPy ``Retrieve``-compatible module.

    Drop this anywhere DSPy expects a retrieval step.  The module queries the
    underlying LangChain retriever and returns a ``dspy.Prediction`` with a
    ``passages`` field (list of strings).

    Args:
        retriever: Any LangChain ``BaseRetriever`` instance.
        k: Default number of passages to retrieve.

    Example:
        >>> from langchain_community.vectorstores import FAISS
        >>> from langchain_openai import OpenAIEmbeddings
        >>> from dspy_lc_bridge import LangChainRetriever
        >>>
        >>> vs = FAISS.load_local("index", OpenAIEmbeddings())
        >>> retriever = LangChainRetriever(vs.as_retriever(), k=3)
        >>>
        >>> class RAG(dspy.Module):
        ...     def __init__(self):
        ...         self.retrieve = retriever
        ...         self.generate = dspy.ChainOfThought("context, question -> answer")
        ...
        ...     def forward(self, question):
        ...         passages = self.retrieve(question).passages
        ...         return self.generate(context="\\n".join(passages), question=question)
    """

    def __init__(self, retriever: Any, k: int = 3) -> None:
        self._retriever = retriever
        self.k = k

    def __call__(
        self,
        query_or_queries: str | list[str],
        k: int | None = None,
    ) -> Any:
        """Retrieve passages for one or more queries.

        Args:
            query_or_queries: A single query string or a list of queries.
            k: Number of results to return per query.  Overrides the
               instance-level default when provided.

        Returns:
            A ``dspy.Prediction`` with a ``passages`` field (list of str).
        """
        import dspy

        effective_k = k if k is not None else self.k

        if isinstance(query_or_queries, str):
            queries = [query_or_queries]
        else:
            queries = query_or_queries

        passages: list[str] = []
        for query in queries:
            docs = self._retriever.invoke(query)
            for doc in docs[:effective_k]:
                passages.append(doc.page_content if hasattr(doc, "page_content") else str(doc))

        return dspy.Prediction(passages=passages)

    async def acall(
        self,
        query_or_queries: str | list[str],
        k: int | None = None,
    ) -> Any:
        """Async version of retrieval.

        Args:
            query_or_queries: Single query or list of queries.
            k: Number of results per query.

        Returns:
            A ``dspy.Prediction`` with a ``passages`` field.
        """
        import dspy

        effective_k = k if k is not None else self.k

        if isinstance(query_or_queries, str):
            queries = [query_or_queries]
        else:
            queries = query_or_queries

        passages: list[str] = []
        for query in queries:
            if hasattr(self._retriever, "ainvoke") and inspect.iscoroutinefunction(
                self._retriever.ainvoke
            ):
                docs = await self._retriever.ainvoke(query)
            else:
                loop = asyncio.get_event_loop()
                docs = await loop.run_in_executor(None, self._retriever.invoke, query)
            for doc in docs[:effective_k]:
                passages.append(doc.page_content if hasattr(doc, "page_content") else str(doc))

        return dspy.Prediction(passages=passages)

    def __repr__(self) -> str:
        return f"LangChainRetriever(retriever={self._retriever.__class__.__name__}, k={self.k})"


# ---------------------------------------------------------------------------
# Bridge 5: DSPy Retrieve → LangChain BaseRetriever
# ---------------------------------------------------------------------------


class DSPyRetriever:
    """Exposes a DSPy ``Retrieve`` module as a LangChain ``BaseRetriever``.

    After wrapping, the object can be used anywhere LangChain expects a
    retriever — e.g. ``RetrievalQA``, ``ConversationalRetrievalChain``, or
    ``create_retrieval_chain``.

    Args:
        dspy_retrieve: An instantiated DSPy ``Retrieve`` module (or any
                       callable that accepts a query string and returns a
                       ``Prediction`` with a ``passages`` field).

    Example:
        >>> import dspy
        >>> from dspy_lc_bridge import DSPyRetriever
        >>>
        >>> dspy_rm = dspy.Retrieve(k=5)
        >>> retriever = DSPyRetriever(dspy_rm)
        >>>
        >>> from langchain.chains import RetrievalQA
        >>> qa = RetrievalQA.from_chain_type(llm=my_llm, retriever=retriever)
    """

    def __init__(self, dspy_retrieve: Any) -> None:
        self._dspy_retrieve = dspy_retrieve

    # ------------------------------------------------------------------
    # LangChain BaseRetriever protocol (duck-typed for compatibility)
    # ------------------------------------------------------------------

    def invoke(self, query: str, **kwargs: Any) -> list[Any]:
        """Retrieve documents for *query*.

        Args:
            query: The search query string.

        Returns:
            A list of LangChain ``Document`` objects.
        """
        return self._get_relevant_documents(query)

    def _get_relevant_documents(self, query: str, **kwargs: Any) -> list[Any]:
        from langchain_core.documents import Document

        prediction = self._dspy_retrieve(query)
        passages = getattr(prediction, "passages", [])
        return [
            Document(
                page_content=p if isinstance(p, str) else getattr(p, "long_text", str(p)),
                metadata={"source": "dspy"},
            )
            for p in passages
        ]

    async def ainvoke(self, query: str, **kwargs: Any) -> list[Any]:
        """Async retrieval — wraps the synchronous DSPy call in an executor.

        Args:
            query: The search query string.

        Returns:
            A list of LangChain ``Document`` objects.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_relevant_documents, query)

    async def _aget_relevant_documents(self, query: str, **kwargs: Any) -> list[Any]:
        return await self.ainvoke(query)

    def get_relevant_documents(self, query: str, **kwargs: Any) -> list[Any]:
        """LangChain legacy retriever interface."""
        return self._get_relevant_documents(query)

    def __repr__(self) -> str:
        return f"DSPyRetriever(dspy_retrieve={self._dspy_retrieve.__class__.__name__})"
