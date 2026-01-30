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

def test_qa_endpoint_invalid_questions_json():
    resp = client.post(
        "/qa",
        files={
            "questions_file": ("questions.json", "{bad json", "application/json"),
            "document_file": ("doc.json", '{"cloud":"AWS"}', "application/json"),
        },
    )
    assert resp.status_code == 400


def test_qa_endpoint_too_many_questions(monkeypatch):
    # set max_questions small by mocking get_settings()
    from app.core import config as config_module

    real = config_module.get_settings()
    real.max_questions = 1

    def fake_settings():
        return real

    monkeypatch.setattr("app.controllers.qa_controller.get_settings", fake_settings)

    resp = client.post(
        "/qa",
        files={
            "questions_file": ("questions.json", '["Q1","Q2"]', "application/json"),
            "document_file": ("doc.json", '{"cloud":"AWS"}', "application/json"),
        },
    )
    assert resp.status_code == 400


def test_qa_endpoint_doc_too_large(monkeypatch):
    from app.core import config as config_module

    real = config_module.get_settings()
    real.max_doc_bytes = 5

    def fake_settings():
        return real

    monkeypatch.setattr("app.controllers.qa_controller.get_settings", fake_settings)

    resp = client.post(
        "/qa",
        files={
            "questions_file": ("questions.json", '["Q1"]', "application/json"),
            "document_file": ("doc.json", "0123456789", "application/json"),
        },
    )
    assert resp.status_code == 413
