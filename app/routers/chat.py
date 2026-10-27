from datetime import date
from fastapi import APIRouter, Depends, HTTPException
from openai import APIConnectionError, NotFoundError
from sqlalchemy.orm import Session
from app.config import get_settings
from app.database import get_db
from app.models.db_models import ChatMessageLog
from app.models.schemas import ChatRequest, ChatResponse
from app.services.booking_service import BookingValidationError, save_booking
from app.services.llm_service import LLMService
from app.services.redis_client import get_memory
from app.services.vector_store import get_vector_store


router = APIRouter(prefix="/api/v1", tags=["Conversational RAG"])


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, db: Session = Depends(get_db)) -> ChatResponse:
    memory = get_memory()
    llm = LLMService()
    try:
        history = memory.get_history(request.session_id)
        query_embedding = llm.embed_texts([request.message])[0]
        settings_top_k = get_settings().retrieval_top_k
        chunks = get_vector_store().search(query_embedding, limit=settings_top_k)

        answer = llm.answer(request.message, chunks, history)
        booking_created = False
        booking = None

        # Try booking extraction only when the user asks to book now,
        # or when the assistant's last reply was asking for missing booking details.
        booking_terms = ("book", "schedule", "appointment")
        wants_booking = any(term in request.message.lower() for term in booking_terms)
        booking_in_progress = bool(history) and "To complete the booking" in history[-1]["content"]
        if wants_booking or booking_in_progress:
            details = llm.extract_booking_details(request.message, history)
            fields_present = all([
                details.name, details.email, details.interview_date, details.interview_time
            ])
            if fields_present:
                try:
                    booking = save_booking(db, details)
                    booking_created = True
                    answer = (
                        f"Your interview booking has been saved as pending. "
                        f"Name: {booking.name}; email: {booking.email}; "
                        f"date: {booking.interview_date}; time: {booking.interview_time}."
                    )
                except BookingValidationError as exc:
                    answer = f"I couldn't save the booking: {exc}. Please provide corrected details."
            else:
                missing = []
                if not details.name: missing.append("full name")
                if not details.email: missing.append("email address")
                if not details.interview_date: missing.append("interview date")
                if not details.interview_time: missing.append("interview time")
                answer += "\n\nTo complete the booking, please provide: " + ", ".join(missing) + "."

        memory.add_message(request.session_id, "user", request.message)
        memory.add_message(request.session_id, "assistant", answer)
        db.add_all([
            ChatMessageLog(session_id=request.session_id, role="user", content=request.message),
            ChatMessageLog(session_id=request.session_id, role="assistant", content=answer),
        ])
        db.commit()
        return ChatResponse(
            session_id=request.session_id,
            answer=answer,
            booking_created=booking_created,
            booking=booking,
            retrieved_chunks=len(chunks),
        )
    except HTTPException:
        raise
    except APIConnectionError as exc:
        db.rollback()
        raise HTTPException(status_code=503, detail="Cannot reach the model server. Is Ollama running?") from exc
    except NotFoundError as exc:
        db.rollback()
        raise HTTPException(status_code=503, detail="Model not found. Run 'ollama pull' for the models in .env") from exc
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=502, detail="Chat request failed; check service configuration and logs") from exc
