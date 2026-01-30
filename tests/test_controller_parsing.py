import json

import pytest

from app.controllers.qa_controller import parse_questions, parse_document


class DummyUploadFile:
    # minimal stand-in to pass filename/content_type
    def __init__(self, filename: str, content_type: str):
        self.filename = filename
        self.content_type = content_type


@pytest.mark.anyio
async def test_parse_questions_invalid_json():
    with pytest.raises(Exception) as e:
        await parse_questions(b"{not json", max_questions=50)
    assert "Invalid questions_file" in str(e.value)


@pytest.mark.anyio
async def test_parse_questions_not_list():
    payload = json.dumps({"q": "x"}).encode("utf-8")
    with pytest.raises(Exception) as e:
        await parse_questions(payload, max_questions=50)
    assert "Invalid questions_file" in str(e.value)


@pytest.mark.anyio
async def test_parse_questions_too_many():
    payload = json.dumps(["q"] * 3).encode("utf-8")
    with pytest.raises(Exception) as e:
        await parse_questions(payload, max_questions=2)
    assert "Too many questions" in str(e.value)


@pytest.mark.anyio
async def test_parse_document_too_large():
    dummy_file = DummyUploadFile(filename="doc.json", content_type="application/json")
    with pytest.raises(Exception) as e:
        await parse_document(b"x" * 11, dummy_file, max_doc_bytes=10)
    assert "Document too large" in str(e.value)


@pytest.mark.anyio
async def test_parse_document_unsupported_type():
    dummy_file = DummyUploadFile(filename="doc.txt", content_type="text/plain")
    with pytest.raises(Exception) as e:
        await parse_document(b"hello", dummy_file, max_doc_bytes=10_000_000)
    assert "Invalid document_file" in str(e.value)
