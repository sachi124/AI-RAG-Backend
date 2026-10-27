from io import BytesIO
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session
import pymupdf
from openai import APIConnectionError, NotFoundError
from app.config import get_settings
from app.database import get_db
from app.models.schemas import IngestionResponse
from app.services.document_service import DocumentService
from app.services.llm_service import LLMService
from app.services.vector_store import get_vector_store
import logging


router = APIRouter(prefix="/api/v1/documents", tags=["Document ingestion"])
ALLOWED_EXTENSIONS = {".pdf", ".txt"}


@router.post("/ingest", response_model=IngestionResponse, status_code=status.HTTP_201_CREATED)
async def ingest_document(
    file: UploadFile = File(...),
    chunking_strategy: str = Form("recursive", pattern="^(fixed|recursive)$"),
    db: Session = Depends(get_db),
) -> IngestionResponse:
    settings = get_settings()
    filename = file.filename or "uploaded_file"
    extension = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=415, detail="Only PDF and TXT files are supported")

    raw = await file.read()
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if len(raw) > max_bytes:
        raise HTTPException(status_code=413, detail=f"File exceeds {settings.max_upload_size_mb} MB limit")

    try:
        if extension == ".txt":
            text = raw.decode("utf-8-sig")
        else:
            with pymupdf.open(stream=BytesIO(raw), filetype="pdf") as pdf:
                text = "\n\n".join(page.get_text("text") for page in pdf)
    except (UnicodeDecodeError, pymupdf.FileDataError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail="Could not read this document") from exc

    try:
        service = DocumentService(LLMService(), get_vector_store())
        document_id, count = service.ingest(db, filename, file.content_type or "application/octet-stream",
                                            text, chunking_strategy)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except APIConnectionError as exc:
        db.rollback()
        raise HTTPException(status_code=503, detail="Cannot reach the model server. Is Ollama running?") from exc
    except NotFoundError as exc:
        db.rollback()
        raise HTTPException(status_code=503, detail=f"Model not found. Pull it first, e.g. 'ollama pull {settings.openai_embedding_model}'") from exc
    except Exception as exc:
        logging.exception("Document ingestion failed")
        db.rollback()
        # Keep internal exception details out of public API responses.
        raise HTTPException(status_code=502, detail="Document ingestion failed; check service configuration and logs") from exc

    return IngestionResponse(
        document_id=document_id,
        filename=filename,
        chunking_strategy=chunking_strategy,  # type: ignore[arg-type]
        chunk_count=count,
        message="Document ingested successfully",
    )
