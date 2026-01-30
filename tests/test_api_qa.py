from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_qa_endpoint(monkeypatch):
    def fake_run(self, questions, document_text):
        return [
            {"question": questions[0], "answer": "AWS", "sources": ["page 1"]}
        ]

    monkeypatch.setattr(
        "app.services.qa_service.QAService.run",
        fake_run,
    )

    response = client.post(
        "/qa",
        files={
            "questions_file": ("questions.json", '["Which cloud?"]', "application/json"),
            "document_file": ("doc.json", '{"cloud":"AWS"}', "application/json"),
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["results"][0]["answer"] == "AWS"
