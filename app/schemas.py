from typing import List
from pydantic import BaseModel


class QAResult(BaseModel):
    question: str
    answer: str
    sources: List[str] = []


class QAResponse(BaseModel):
    results: List[QAResult]
