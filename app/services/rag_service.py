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

_SYSTEM = """You are a helpful question answering assistant.
Use the provided context to answer the question accurately and concisely.

IMPORTANT RULES:
1. Base your answer on the context provided
2. If the context contains relevant information, answer the question even if it's not a perfect match
3. If the answer is clearly not in the context, respond with exactly: Not found
4. Keep answers concise but complete
5. You can paraphrase or synthesize information from the context
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
        
        # Parse document text to extract page information
        docs = self._parse_document_to_docs(document_text)
        
        # Split documents while preserving metadata
        chunks = self._splitter.split_documents(docs)
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

        # Use async embedding generation with automatic batching
        vectorstore = await asyncio.to_thread(
            FAISS.from_documents,
            documents=chunks,
            embedding=self._embeddings,
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
        search_kwargs = {
            "k": self.settings.retrieval_k,
        }
        
        # Only add fetch_k for MMR (it needs a larger pool to select from)
        if self.settings.use_mmr:
            search_kwargs["fetch_k"] = self.settings.retrieval_k * 3
        
        retriever = vectorstore.as_retriever(
            search_kwargs=search_kwargs,
            search_type="mmr" if self.settings.use_mmr else "similarity",
        )

        docs: List[Document] = await retriever.ainvoke(question)
        
        logger.info(
            "rag_documents_retrieved",
            extra={
                "extra": {
                    "question": question[:100],
                    "num_docs": len(docs),
                    "search_type": "mmr" if self.settings.use_mmr else "similarity",
                    "k": self.settings.retrieval_k,
                }
            },
        )
        
        # Early exit if no relevant docs found
        if not docs:
            logger.warning(
                "rag_no_documents_found",
                extra={"extra": {"question": question[:100]}}
            )
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
        
        # Only include sources if an actual answer was found
        sources = self._extract_sources(docs) if answer != "Not found" else []

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
    def _parse_document_to_docs(document_text: str) -> List[Document]:
        """Parse document text into Document objects with page metadata."""
        docs = []
        sections = document_text.split("\n\n")
        current_page = None
        current_content = []
        
        for section in sections:
            section = section.strip()
            if not section:
                continue
            
            # Check if this section starts with a page marker
            if section.startswith("[page "):
                # Save previous page if exists
                if current_page and current_content:
                    docs.append(Document(
                        page_content="\n\n".join(current_content),
                        metadata={"source": current_page}
                    ))
                    current_content = []
                
                # Extract page number and content
                lines = section.split("\n", 1)
                current_page = lines[0].strip("[]")
                if len(lines) > 1:
                    current_content.append(lines[1])
            else:
                # Continue adding to current page
                current_content.append(section)
        
        # Add the last page
        if current_page and current_content:
            docs.append(Document(
                page_content="\n\n".join(current_content),
                metadata={"source": current_page}
            ))
        
        # If no page markers found, treat entire text as single document
        if not docs:
            docs.append(Document(
                page_content=document_text,
                metadata={"source": "document"}
            ))
        
        return docs
    
    @staticmethod
    def _get_cache_key(question: str) -> str:
        """Generate a cache key for a question."""
        return hashlib.sha256(question.encode()).hexdigest()
    
    @staticmethod
    def _extract_sources(docs: List[Document]) -> List[str]:
        sources = []
        for d in docs:
            # First try to get source from metadata
            if d.metadata and "source" in d.metadata:
                sources.append(d.metadata["source"])
            else:
                # Fallback to parsing from content
                first_line = (d.page_content.splitlines() or [""])[0].strip()
                if first_line.startswith("[page "):
                    sources.append(first_line.strip("[]"))
        return sorted(set(sources))
