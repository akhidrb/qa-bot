from dataclasses import dataclass
from typing import Any, Dict, List

from app.services.rag_service import RAGService


@dataclass
class QAService:
    rag_service: RAGService

    def run(
            self,
            *,
            questions: List[str],
            document_text: str,
    ) -> List[Dict[str, Any]]:
        """
        Orchestrates QA over an already-parsed document.
        Assumes inputs are validated.
        """
        if not document_text.strip():
            raise ValueError("Document text is empty")

        if not questions:
            raise ValueError("No questions provided")

        vectorstore = self.rag_service.build_index(document_text)
        return self.rag_service.answer_many(vectorstore, questions)
