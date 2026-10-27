from datetime import date, datetime, time
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from app.models.db_models import InterviewBooking
from app.models.schemas import BookingDetails, BookingResponse


class BookingValidationError(ValueError):
    pass


def save_booking(db: Session, details: BookingDetails) -> BookingResponse:
    missing = [
        field for field, value in {
            "name": details.name,
            "email": details.email,
            "interview_date": details.interview_date,
            "interview_time": details.interview_time,
        }.items() if value is None
    ]
    if missing:
        raise BookingValidationError(f"Missing booking fields: {', '.join(missing)}")
    if not details.name or not details.name.strip():
        raise BookingValidationError("Name cannot be empty")
    if details.interview_date < date.today():
        raise BookingValidationError("Interview date must not be in the past")

    booking = InterviewBooking(
        name=details.name.strip(),
        email=str(details.email),
        interview_date=details.interview_date.isoformat(),
        interview_time=details.interview_time.strftime("%H:%M"),
        status="pending",
    )
    db.add(booking)
    try:
        db.commit()
        db.refresh(booking)
    except IntegrityError as exc:
        db.rollback()
        raise BookingValidationError("A booking already exists for this email, date, and time") from exc

    return BookingResponse(
        id=booking.id,
        name=booking.name,
        email=booking.email,
        interview_date=date.fromisoformat(booking.interview_date),
        interview_time=time.fromisoformat(booking.interview_time),
        status=booking.status,
    )
