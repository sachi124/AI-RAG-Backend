from collections.abc import Sequence


def fixed_size_chunks(text: str, chunk_size: int = 800, overlap: int = 120) -> list[str]:
    """Split text into fixed character windows with overlap."""
    text = text.strip()
    if not text:
        return []
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")
    step = chunk_size - overlap
    return [text[start:start + chunk_size].strip() for start in range(0, len(text), step)
            if text[start:start + chunk_size].strip()]


def recursive_chunks(text: str, chunk_size: int = 800, overlap: int = 120) -> list[str]:
    """
    Prefer paragraph boundaries, then line/sentence/space boundaries.
    A small overlap is added between final chunks to preserve context.
    """
    text = text.strip()
    if not text:
        return []
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    separators = ["\n\n", "\n", ". ", " ", ""]
    pieces = [text]
    for sep in separators:
        next_pieces: list[str] = []
        for piece in pieces:
            if len(piece) <= chunk_size:
                next_pieces.append(piece)
                continue
            if sep == "":
                next_pieces.extend(piece[i:i + chunk_size] for i in range(0, len(piece), chunk_size))
            else:
                parts = piece.split(sep)
                current = ""
                for part in parts:
                    candidate = part if not current else current + sep + part
                    if len(candidate) <= chunk_size:
                        current = candidate
                    else:
                        if current:
                            next_pieces.append(current)
                        if len(part) > chunk_size:
                            next_pieces.append(part)
                            current = ""
                        else:
                            current = part
                if current:
                    next_pieces.append(current)
        pieces = next_pieces

    # Merge small pieces where possible, preserving a practical chunk size.
    merged: list[str] = []
    current = ""
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        candidate = f"{current}\n{piece}" if current else piece
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                merged.append(current)
            current = piece
    if current:
        merged.append(current)

    if overlap == 0 or len(merged) <= 1:
        return merged

    result: list[str] = []
    previous = ""
    for chunk in merged:
        if previous:
            prefix = previous[-overlap:]
            chunk = f"{prefix}{chunk}"
            chunk = chunk[-chunk_size:]
        result.append(chunk)
        previous = chunk
    return result


def split_text(text: str, strategy: str, chunk_size: int, overlap: int) -> list[str]:
    if strategy == "fixed":
        return fixed_size_chunks(text, chunk_size, overlap)
    if strategy == "recursive":
        return recursive_chunks(text, chunk_size, overlap)
    raise ValueError("strategy must be 'fixed' or 'recursive'")
