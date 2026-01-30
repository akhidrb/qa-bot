import pytest
from app.services.qa_service import QAService


class FakeRAGService:
    def __init__(self):
        self.called_with = None

    def build_index(self, document_text: str):
        assert document_text == "doc text"
        return "VECTORSTORE"

    def answer_many(self, vectorstore, questions):
        self.called_with = (vectorstore, questions)
        return [
            {"question": "Q1", "answer": "A1", "sources": ["page 1"]},
        ]


def test_qa_service_happy_path():
    rag = FakeRAGService()
    service = QAService(rag_service=rag)

    result = service.run(
        questions=["Q1"],
        document_text="doc text",
    )

    assert rag.called_with == ("VECTORSTORE", ["Q1"])
    assert result == [
        {"question": "Q1", "answer": "A1", "sources": ["page 1"]},
    ]


def test_qa_service_empty_document():
    rag = FakeRAGService()
    service = QAService(rag_service=rag)

    with pytest.raises(ValueError, match="Document text is empty"):
        service.run(
            questions=["Q1"],
            document_text="   ",
        )


def test_qa_service_no_questions():
    rag = FakeRAGService()
    service = QAService(rag_service=rag)

    with pytest.raises(ValueError, match="No questions provided"):
        service.run(
            questions=[],
            document_text="doc text",
        )
