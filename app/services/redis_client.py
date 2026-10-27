import json
from typing import Any
from redis import Redis
from app.config import get_settings


class RedisMemory:
    def __init__(self) -> None:
        settings = get_settings()
        self.client: Redis = Redis.from_url(settings.redis_url, decode_responses=True)
        self.ttl = settings.redis_memory_ttl_seconds

    def get_history(self, session_id: str) -> list[dict[str, str]]:
        key = self._key(session_id)
        raw_items = self.client.lrange(key, 0, -1)
        history: list[dict[str, str]] = []
        for item in raw_items:
            try:
                decoded: Any = json.loads(item)
                if decoded.get("role") in {"user", "assistant"} and isinstance(decoded.get("content"), str):
                    history.append({"role": decoded["role"], "content": decoded["content"]})
            except (json.JSONDecodeError, AttributeError):
                continue
        return history

    def add_message(self, session_id: str, role: str, content: str) -> None:
        if role not in {"user", "assistant"}:
            raise ValueError("Only user and assistant messages may be stored")
        key = self._key(session_id)
        self.client.rpush(key, json.dumps({"role": role, "content": content}))
        self.client.expire(key, self.ttl)
        # Keep memory bounded. A pair of messages is one conversation turn.
        self.client.ltrim(key, -40, -1)

    @staticmethod
    def _key(session_id: str) -> str:
        return f"palm_mind:chat:{session_id}"


_memory: RedisMemory | None = None


def get_memory() -> RedisMemory:
    global _memory
    if _memory is None:
        _memory = RedisMemory()
    return _memory
