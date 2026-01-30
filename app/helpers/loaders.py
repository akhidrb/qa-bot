import json
from pypdf import PdfReader
from io import BytesIO


def load_document_text(filename: str, content_type: str, raw_bytes: bytes) -> str:
    name = filename.lower()
    if name.endswith(".pdf") or content_type == "application/pdf":
        return _load_pdf(raw_bytes)
    return _load_json(raw_bytes)


def _load_pdf(pdf_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(pdf_bytes))
    parts = []
    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if text:
            parts.append(f"[page {i+1}]\n{text}")
    return "\n\n".join(parts)


def _load_json(json_bytes: bytes) -> str:
    obj = json.loads(json_bytes.decode("utf-8"))
    return json.dumps(obj, ensure_ascii=False, indent=2)
