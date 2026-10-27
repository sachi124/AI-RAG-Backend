
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.database import init_db
from app.models.schemas import HealthResponse
from app.routers import chat, ingestion


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="Palm Mind AI RAG Backend",
    description=(
        "Document ingestion, custom conversational RAG, "
        "Redis memory, and interview booking."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://127.0.0.1"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(ingestion.router)
app.include_router(chat.router)


@app.get("/", tags=["Health"])
def root() -> dict[str, str]:
    return {"message": "Palm Mind AI RAG backend is running"}


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health() -> HealthResponse:
    return HealthResponse(status="ok")
