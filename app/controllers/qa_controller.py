import asyncio
import json
import logging
import time

from fastapi import UploadFile, HTTPException
from app.core.config import get_settings
from app.schemas import QAResponse, QAResult
from app.helpers.loaders import load_document_text
from app.services.qa_service import QAService
from app.services.rag_service import RAGService

logger = logging.getLogger("app.qa")


async def handle_qa(questions_file: UploadFile, document_file: UploadFile) -> QAResponse:
    settings = get_settings()
    t0 = time.time()

    # Read files concurrently for efficiency
    q_bytes, d_bytes = await asyncio.gather(
        questions_file.read(),
        document_file.read(),
    )

    logger.info(
        "qa_upload_received",
        extra={
            "extra": {
                "q_bytes": len(q_bytes),
                "d_bytes": len(d_bytes),
                "questions_filename": questions_file.filename,
                "document_filename": document_file.filename,
                "document_type": document_file.content_type,
            }
        },
    )

    # Parse questions and document concurrently
    questions_obj, doc_text = await asyncio.gather(
        parse_questions(q_bytes, settings.max_questions),
        parse_document(d_bytes, document_file, settings.max_doc_bytes),
    )

    logger.info(
        "qa_inputs_parsed",
        extra={
            "extra": {
                "questions_count": len(questions_obj),
                "doc_text_len": len(doc_text),
            }
        },
    )

    rag_service = RAGService(settings=settings)
    qa_service = QAService(rag_service=rag_service)

    results = await qa_service.run(
        questions=questions_obj,
        document_text=doc_text,
    )

    logger.info(
        "qa_completed",
        extra={
            "extra": {
                "questions_count": len(questions_obj),
                "results_count": len(results),
                "total_ms": int((time.time() - t0) * 1000),
            }
        },
    )
    return QAResponse(results=[QAResult(**r) for r in results])


async def parse_document(d_bytes: bytes, document_file: UploadFile, max_doc_bytes: int):
    if len(d_bytes) > max_doc_bytes:
        raise HTTPException(status_code=413, detail="Document too large")
    try:
        doc_text = load_document_text(
            filename=document_file.filename or "",
            content_type=document_file.content_type or "",
            raw_bytes=d_bytes,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid document_file: {e}")
    if not doc_text.strip():
        raise HTTPException(status_code=400, detail="Document is empty or unreadable")
    return doc_text


async def parse_questions(q_bytes: bytes, max_questions: int):
    try:
        questions_obj = json.loads(q_bytes.decode("utf-8"))
        if not isinstance(questions_obj, list) or not all(isinstance(x, str) for x in questions_obj):
            raise ValueError("questions JSON must be a list of strings")
        if len(questions_obj) > max_questions:
            raise ValueError(f"Too many questions (max {max_questions})")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid questions_file: {e}")
    return questions_obj
