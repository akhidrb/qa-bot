from pathlib import Path
from fastapi.responses import FileResponse
from fastapi import APIRouter, UploadFile, File
from app.controllers.qa_controller import handle_qa
from app.schemas import QAResponse

router = APIRouter()


@router.get("/", include_in_schema=False)
def index():
    path = Path(__file__).resolve().parent / "static" / "index.html"
    return FileResponse(path)


@router.post("/qa", response_model=QAResponse)
async def qa(
        questions_file: UploadFile = File(...),
        document_file: UploadFile = File(...),
) -> QAResponse:
    return await handle_qa(questions_file, document_file)
