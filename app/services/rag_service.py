from __future__ import annotations

import asyncio
import hashlib
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

        # Use batching for efficient embeddings
        self._embeddings = OpenAIEmbeddings(
            model=self.settings.embedding_model,
            chunk_size=self.settings.embedding_batch_size,
        )

        self._llm = ChatOpenAI(
            model=self.settings.openai_model,
            temperature=0,
            timeout=settings.openai_timeout,
        )
        
        # Cache for deduplicating identical questions
        self._answer_cache: Dict[str, Dict[str, Any]] = {}

    async def build_index(self, document_text: str) -> FAISS:
        """Async index building with batched embeddings for efficiency."""
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

        # Use async embedding generation with automatic batching
        vectorstore = await asyncio.to_thread(
            FAISS.from_texts,
            texts=chunks,
            embedding=self._embeddings,
            metadatas=meta_datas,
        )

        logger.info(
            "rag_index_built",
            extra={
                "extra": {
                    "chunks": len(chunks),
                    "build_ms": int((time.time() - t0) * 1000),
                }
            },
        )
        return vectorstore

    async def answer_many(
            self, vectorstore: FAISS, questions: List[str]
    ) -> List[Dict[str, Any]]:
        """Process multiple questions concurrently with deduplication and batching."""
        if not questions:
            return []
        
        # Deduplicate questions to avoid redundant LLM calls
        unique_questions = list(dict.fromkeys(questions))  # Preserves order
        logger.info(
            "rag_batch_processing",
            extra={
                "extra": {
                    "total_questions": len(questions),
                    "unique_questions": len(unique_questions),
                    "duplicates_eliminated": len(questions) - len(unique_questions),
                }
            },
        )
        
        # Process all unique questions concurrently
        tasks = [
            self.answer_one(vectorstore, q)
            for q in unique_questions
        ]
        unique_results = await asyncio.gather(*tasks)
        
        # Map results back to original question list (including duplicates)
        result_map = {r["question"]: r for r in unique_results}
        return [result_map[q] for q in questions]

    async def answer_one(
            self, vectorstore: FAISS, question: str
    ) -> Dict[str, Any]:
        """Answer a single question with caching and async execution."""
        t0 = time.time()
        
        # Check cache first to avoid unnecessary LLM calls
        cache_key = self._get_cache_key(question)
        if cache_key in self._answer_cache:
            logger.info(
                "rag_cache_hit",
                extra={"extra": {"question_len": len(question)}}
            )
            return self._answer_cache[cache_key]
        
        # Retrieve relevant documents (async to not block)
        retriever = vectorstore.as_retriever(
            search_kwargs={
                "k": self.settings.retrieval_k,
                "fetch_k": self.settings.retrieval_k * 2,  # Fetch more for MMR
            },
            search_type="mmr" if self.settings.use_mmr else "similarity",
        )

        docs: List[Document] = await retriever.ainvoke(question)
        
        # Early exit if no relevant docs found
        if not docs:
            result = {
                "question": question,
                "answer": "Not found",
                "sources": [],
            }
            self._answer_cache[cache_key] = result
            return result
        
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
        # Use async invoke to not block event loop
        response = await self._llm.ainvoke(msg)
        answer = response.content.strip()
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

        result = {
            "question": question,
            "answer": answer,
            "sources": sources,
        }
        
        # Cache the result
        self._answer_cache[cache_key] = result
        return result

    @staticmethod
    def _get_cache_key(question: str) -> str:
        """Generate a cache key for a question."""
        return hashlib.sha256(question.encode()).hexdigest()
    
    @staticmethod
    def _extract_sources(docs: List[Document]) -> List[str]:
        sources = []
        for d in docs:
            first_line = (d.page_content.splitlines() or [""])[0].strip()
            if first_line.startswith("[page "):
                sources.append(first_line.strip("[]"))
        return sorted(set(sources))
