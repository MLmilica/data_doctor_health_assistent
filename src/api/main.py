"""FastAPI entry point."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.dependencies import shutdown, startup
from api.routes import chat, health


@asynccontextmanager
async def lifespan(_: FastAPI):
    startup()
    try:
        yield
    finally:
        shutdown()


app = FastAPI(
    title="Data Doctor API",
    description="Clinical analytics prototype — prediction slice.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(chat.router)
