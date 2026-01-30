from fastapi import APIRouter, UploadFile, File
from app.controllers.qa_controller import handle_qa
from app.schemas import QAResponse

router = APIRouter()


@router.post("/qa", response_model=QAResponse)
async def qa(
        questions_file: UploadFile = File(...),
        document_file: UploadFile = File(...),
) -> QAResponse:
    return await handle_qa(questions_file, document_file)
