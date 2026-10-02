"""FastAPI app factory. Run it with:  uvicorn app:app --reload   (see app.py)  ->  http://localhost:8000"""
from __future__ import annotations

import base64
import logging
import os
import re
import threading
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from . import clients
from .agent import Agent
from .clients import ServiceError
from .guardrails import redact
from .tools import ToolBox, load_embedder

ROOT = Path(__file__).parent.parent
STATIC = ROOT / "static"
log = logging.getLogger("voicedesk")
SESSION_TTL = 30 * 60
MAX_SESSIONS = 200
RATE_LIMIT = 30  # requests per minute per client
SESSION_ID = re.compile(r"^[A-Za-z0-9_-]{8,64}$")


class RateLimiter:
    """Sliding-window limiter: protects free-tier API credits from a runaway client."""

    def __init__(self, per_minute: int, clock=time.monotonic):
        self.per_minute, self.clock, self.hits, self.lock = per_minute, clock, {}, threading.Lock()

    def allow(self, key: str) -> bool:
        now = self.clock()
        with self.lock:
            window = [t for t in self.hits.get(key, []) if now - t < 60]
            if len(window) >= self.per_minute:
                self.hits[key] = window
                return False
            window.append(now)
            self.hits[key] = window
            return True


class Sessions:
    def __init__(self, ttl: int = SESSION_TTL, max_sessions: int = MAX_SESSIONS, clock=time.monotonic):
        self.ttl, self.max, self.clock, self.items, self.lock = ttl, max_sessions, clock, {}, threading.Lock()

    def get(self, session_id: str) -> list[dict]:
        now = self.clock()
        with self.lock:
            for sid in [s for s, (_, seen) in self.items.items() if now - seen > self.ttl]:
                del self.items[sid]
            if session_id not in self.items and len(self.items) >= self.max:
                oldest = min(self.items, key=lambda s: self.items[s][1])
                del self.items[oldest]
            history = self.items.get(session_id, ([], now))[0]
            self.items[session_id] = (history, now)
            return history


class ChatRequest(BaseModel):
    session_id: str
    text: str = Field(min_length=1, max_length=1000)


def create_app(llm=None, stt=None, tts=None, toolbox: ToolBox | None = None, rate_limit: int = RATE_LIMIT) -> FastAPI:
    """Build the app. All three services are injectable so tests (and demos) can run without any API key."""
    load_dotenv(ROOT / ".env")  # project-local only
    app = FastAPI(title="VoiceDesk API", version="1.0.0")
    if toolbox is None:
        embedder = load_embedder() if os.getenv("FAQ_EMBEDDINGS", "auto") != "0" else None  # optional semantic FAQ search
        toolbox = ToolBox(db_path=ROOT / "data" / "callbacks.db", embedder=embedder)
    agent = Agent(llm or clients.chat_with_tools, toolbox)
    sessions, limiter = Sessions(), RateLimiter(rate_limit)
    transcribe = stt or clients.transcribe
    tts_holder = {"tts": tts}

    def speaker():
        if tts_holder["tts"] is None:
            tts_holder["tts"] = clients.get_tts()
        return tts_holder["tts"]

    def guard(request: Request, session_id: str) -> list[dict]:
        if not SESSION_ID.match(session_id or ""):
            raise HTTPException(400, "session_id must be 8-64 characters: letters, digits, - or _")
        if not limiter.allow(request.client.host if request.client else "unknown"):
            raise HTTPException(429, "Too many requests. Please slow down.")
        return sessions.get(session_id)

    def respond(history: list[dict], text: str, extra: dict) -> dict:
        started = time.perf_counter()
        result = agent.turn(text, history)
        payload = {"reply": result.reply, "tool_calls": result.tool_events, "escalated": result.escalated,
                   "audio_base64": None, "audio_mime": "audio/mpeg", "tts_error": None,
                   "transcript": extra["transcript"]}
        tts_started = time.perf_counter()
        try:
            payload["audio_base64"] = base64.b64encode(speaker().synthesize(result.reply)).decode()
        except ServiceError as exc:  # degrade gracefully: the caller still gets the text reply
            payload["tts_error"] = str(exc)
        stt_ms = extra.get("stt_ms", 0)
        payload["metrics_ms"] = {
            "stt": stt_ms,
            "llm": round(result.llm_ms),
            "tts": round((time.perf_counter() - tts_started) * 1000),
            "total": stt_ms + round((time.perf_counter() - started) * 1000),
        }
        log.info("turn | user=%s | reply=%s | tools=%s", redact(text), redact(result.reply), [e["name"] for e in result.tool_events])
        return payload

    @app.exception_handler(ServiceError)
    async def service_error(_request, exc: ServiceError):
        return JSONResponse(status_code=503, content={"detail": str(exc), "service": exc.service})

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/health")
    def health() -> dict:
        provider = getattr(tts, "name", None) or ("elevenlabs" if os.getenv("ELEVENLABS_API_KEY") else "edge")
        return {"status": "ok", "llm_configured": llm is not None or bool(os.getenv("GROQ_API_KEY")), "tts_provider": provider}

    @app.post("/api/chat")
    def chat(req: ChatRequest, request: Request) -> dict:
        history = guard(request, req.session_id)
        return respond(history, req.text, {"transcript": req.text})

    @app.post("/api/voice")
    def voice(request: Request, audio: UploadFile = File(...), session_id: str = Form(...)) -> dict:
        history = guard(request, session_id)
        data = audio.file.read(clients.MAX_AUDIO_BYTES + 1)
        if len(data) > clients.MAX_AUDIO_BYTES:
            raise HTTPException(413, "Audio too large (5 MB max)")
        started = time.perf_counter()
        transcript = transcribe(data, audio.filename or "audio.webm")
        return respond(history, transcript, {"transcript": transcript, "stt_ms": round((time.perf_counter() - started) * 1000)})

    @app.post("/api/session")
    def new_session() -> dict:
        return {"session_id": uuid.uuid4().hex}

    return app
