from fastapi import FastAPI
from app.api import router as api_router
from dotenv import load_dotenv
from app.core.config import get_settings
from app.core.logging import setup_logging


def create_app() -> FastAPI:
    load_dotenv()
    setup_logging()
    get_settings()
    app = FastAPI(title="QA Bot")
    app.include_router(api_router)
    return app


# For uvicorn: uvicorn app.setup:app --reload
app = create_app()
