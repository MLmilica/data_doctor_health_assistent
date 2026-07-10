"""Tests for in-memory session store."""

from memory.session_store import InMemorySessionStore
from schemas.memory import ChatTurn


def test_get_or_create_and_save_round_trip() -> None:
    store = InMemorySessionStore()
    session = store.get_or_create("sess-1", "user-1")
    session.turns.append(ChatTurn(role="user", content="hello"))
    store.save(session)

    loaded = store.get("sess-1", "user-1")
    assert loaded is not None
    assert len(loaded.turns) == 1
    assert loaded.turns[0].content == "hello"


def test_sessions_isolated_by_user() -> None:
    store = InMemorySessionStore()
    store.get_or_create("sess-1", "user-a")
    assert store.get("sess-1", "user-b") is None


def test_clear_removes_session() -> None:
    store = InMemorySessionStore()
    store.get_or_create("sess-1", "user-1")
    store.clear("sess-1", "user-1")
    assert store.get("sess-1", "user-1") is None
