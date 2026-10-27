from datetime import date, time
from typing import Literal
from pydantic import BaseModel, ConfigDict, EmailStr, Field


ChunkingStrategy = Literal["fixed", "recursive"]


class IngestionResponse(BaseModel):
    document_id: str
    filename: str
    chunking_strategy: ChunkingStrategy
    chunk_count: int
    message: str


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=100)
    message: str = Field(min_length=1, max_length=8000)


class BookingDetails(BaseModel):
    name: str | None = Field(default=None, max_length=200)
    email: EmailStr | None = None
    interview_date: date | None = None
    interview_time: time | None = None


class BookingResponse(BaseModel):
    id: int
    name: str
    email: EmailStr
    interview_date: date
    interview_time: time
    status: str

    model_config = ConfigDict(from_attributes=True)


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    booking_created: bool = False
    booking: BookingResponse | None = None
    retrieved_chunks: int = 0


class HealthResponse(BaseModel):
    status: str
