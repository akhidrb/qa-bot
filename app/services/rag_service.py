from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.core.config import Settings

_SYSTEM = """You are a question answering assistant.
Answer using ONLY the provided context.
If the answer is not present in the context, respond with exactly: Not found.
Keep answers concise.
"""

_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _SYSTEM),
        ("human", "Context:\n{context}\n\nQuestion:\n{question}"),
    ]
)

logger = logging.getLogger("app.rag")


class RAGService:
    """
    Implements the full RAG pipeline:
    chunking → embeddings → FAISS → retrieval → LLM answering
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )

        self._embeddings = OpenAIEmbeddings(
            model=self.settings.embedding_model
        )

        self._llm = ChatOpenAI(
            model=self.settings.openai_model,
            temperature=0,
            timeout=settings.openai_timeout,
        )

    def build_index(self, document_text: str) -> FAISS:
        t0 = time.time()
        chunks = self._splitter.split_text(document_text)
        if not chunks:
            raise ValueError("No chunks created from document")

        logger.info(
            "rag_chunked",
            extra={
                "extra": {
                    "chunks": len(chunks),
                    "chunk_size": self.settings.chunk_size,
                    "chunk_overlap": self.settings.chunk_overlap,
                }
            },
        )

        meta_datas = [{"chunk": i} for i in range(len(chunks))]

        logger.info(
            "rag_index_built",
            extra={
                "extra": {
                    "chunks": len(chunks),
                    "build_ms": int((time.time() - t0) * 1000),
                }
            },
        )
        return FAISS.from_texts(
            texts=chunks,
            embedding=self._embeddings,
            metadatas=meta_datas,
        )

    def answer_many(
            self, vectorstore: FAISS, questions: List[str]
    ) -> List[Dict[str, Any]]:
        return [self.answer_one(vectorstore, q) for q in questions]

    def answer_one(
            self, vectorstore: FAISS, question: str
    ) -> Dict[str, Any]:
        t0 = time.time()
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": self.settings.retrieval_k}
        )

        docs: List[Document] = retriever.get_relevant_documents(question)
        context = "\n\n".join(d.page_content for d in docs).strip()

        logger.info(
            "rag_retrieved",
            extra={
                "extra": {
                    "k": self.settings.retrieval_k,
                    "docs": len(docs),
                    "context_len": len(context),
                    "question_len": len(question),
                    "retrieve_ms": int((time.time() - t0) * 1000),
                }
            },
        )

        msg = _PROMPT.format_messages(
            context=context,
            question=question,
        )

        llm_start = time.time()
        answer = self._llm.invoke(msg).content.strip()
        llm_ms = int((time.time() - llm_start) * 1000)
        sources = self._extract_sources(docs)

        logger.info(
            "rag_answer_generated",
            extra={
                "extra": {
                    "question_len": len(question),
                    "answer_len": len(answer),
                    "answer_not_found": answer == "Not found",
                    "sources_count": len(sources),
                    "retrieved_docs": len(docs),
                    "context_len": len(context),
                    "llm_ms": llm_ms,
                    "total_ms": int((time.time() - t0) * 1000),
                }
            },
        )

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
        }

    @staticmethod
    def _extract_sources(docs: List[Document]) -> List[str]:
        sources = []
        for d in docs:
            first_line = (d.page_content.splitlines() or [""])[0].strip()
            if first_line.startswith("[page "):
                sources.append(first_line.strip("[]"))
        return sorted(set(sources))
