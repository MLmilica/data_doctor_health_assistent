"""In-memory session store for chat history and step ledger."""

from __future__ import annotations

from schemas.memory import ChatSession, utc_now


class InMemorySessionStore:
    """Process-local session storage (POC). Replace with DynamoDB/Redis on AWS."""

    def __init__(self) -> None:
        self._sessions: dict[tuple[str, str], ChatSession] = {}

    def get(self, session_id: str, user_id: str) -> ChatSession | None:
        return self._sessions.get((user_id, session_id))

    def get_or_create(self, session_id: str, user_id: str) -> ChatSession:
        key = (user_id, session_id)
        existing = self._sessions.get(key)
        if existing is not None:
            return existing
        session = ChatSession(session_id=session_id, user_id=user_id)
        self._sessions[key] = session
        return session

    def save(self, session: ChatSession) -> None:
        session.updated_at = utc_now()
        self._sessions[(session.user_id, session.session_id)] = session

    def clear(self, session_id: str, user_id: str) -> None:
        self._sessions.pop((user_id, session_id), None)

    def reset(self) -> None:
        self._sessions.clear()


_store: InMemorySessionStore | None = None


def get_session_store() -> InMemorySessionStore:
    global _store
    if _store is None:
        _store = InMemorySessionStore()
    return _store


def reset_session_store() -> None:
    global _store
    if _store is not None:
        _store.reset()
    _store = None
