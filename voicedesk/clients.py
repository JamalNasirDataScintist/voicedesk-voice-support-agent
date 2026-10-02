"""HTTP clients: Groq chat with tool calling, Groq Whisper speech-to-text, ElevenLabs / Edge text-to-speech."""
from __future__ import annotations

import asyncio
import os
import time
from concurrent.futures import ThreadPoolExecutor

import requests

GROQ_CHAT = "https://api.groq.com/openai/v1/chat/completions"
GROQ_STT = "https://api.groq.com/openai/v1/audio/transcriptions"
ELEVEN_TTS = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
MAX_AUDIO_BYTES = 5 * 1024 * 1024


class ServiceError(RuntimeError):
    """A downstream service failed; ``service`` says which so the API can report it."""

    def __init__(self, service: str, message: str):
        super().__init__(message)
        self.service = service


def _groq_key() -> str:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise ServiceError("llm", "GROQ_API_KEY is not set. Copy .env.example to .env and add a free key from console.groq.com.")
    return key


def chat_with_tools(messages: list[dict], tools: list[dict], temperature: float = 0.2, timeout: int = 60) -> dict:
    """One chat-completion step. Returns the assistant message: {'content': str|None, 'tool_calls': [...]}."""
    resp = requests.post(
        GROQ_CHAT,
        headers={"Authorization": f"Bearer {_groq_key()}"},
        json={"model": os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"), "messages": messages, "tools": tools,
              "tool_choice": "auto", "temperature": temperature},
        timeout=timeout,
    )
    if resp.status_code != 200:
        raise ServiceError("llm", f"Groq chat returned {resp.status_code}: {resp.text[:200]}")
    return resp.json()["choices"][0]["message"]


def transcribe(audio: bytes, filename: str = "audio.webm", language: str | None = None, timeout: int = 60) -> str:
    """Groq-hosted Whisper. Free tier friendly; accepts webm/ogg/wav/mp3/m4a."""
    if not audio:
        raise ServiceError("stt", "Empty audio")
    if len(audio) > MAX_AUDIO_BYTES:
        raise ServiceError("stt", "Audio too large (5 MB max)")
    data = {"model": os.getenv("STT_MODEL", "whisper-large-v3-turbo"), "response_format": "json", "temperature": "0"}
    if language:
        data["language"] = language
    resp = requests.post(GROQ_STT, headers={"Authorization": f"Bearer {_groq_key()}"}, files={"file": (filename, audio)}, data=data, timeout=timeout)
    if resp.status_code != 200:
        raise ServiceError("stt", f"Groq transcription returned {resp.status_code}: {resp.text[:200]}")
    return (resp.json().get("text") or "").strip()


class ElevenLabsTTS:
    """ElevenLabs over plain HTTP. Flash models cost about 0.5 credit per character (free plan is about 10k credits/month)."""

    name = "elevenlabs"

    def __init__(self, api_key: str, voice_id: str = "21m00Tcm4TlvDq8ikWAM", model_id: str = "eleven_flash_v2_5", retries: int = 3, sleep=time.sleep):
        if not api_key:
            raise ServiceError("tts", "ELEVENLABS_API_KEY is not set")
        self.api_key, self.voice_id, self.model_id, self.retries, self._sleep = api_key, voice_id, model_id, retries, sleep

    def synthesize(self, text: str) -> bytes:
        headers = {"xi-api-key": self.api_key, "Content-Type": "application/json", "Accept": "audio/mpeg"}
        body = {"text": text, "model_id": self.model_id, "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}}
        last = ""
        for attempt in range(self.retries):
            resp = requests.post(ELEVEN_TTS.format(voice_id=self.voice_id), params={"output_format": "mp3_44100_64"}, headers=headers, json=body, timeout=30)
            if resp.status_code == 200:
                return resp.content
            last = f"{resp.status_code}: {resp.text[:160]}"
            if resp.status_code == 401:
                raise ServiceError("tts", "ElevenLabs rejected the API key (401)")
            if resp.status_code in (402, 403) or "quota_exceeded" in resp.text:
                raise ServiceError("tts", f"ElevenLabs quota or plan limit reached ({last})")
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < self.retries - 1:
                self._sleep(2**attempt)
                continue
            break
        raise ServiceError("tts", f"ElevenLabs request failed ({last})")


def _run(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


class EdgeTTS:
    """Free Microsoft Edge neural voice (no key). Good fallback for development and demos."""

    name = "edge"

    def __init__(self, voice: str = "en-US-AriaNeural"):
        self.voice = voice

    def synthesize(self, text: str) -> bytes:
        import edge_tts

        async def go() -> bytes:
            out = bytearray()
            async for chunk in edge_tts.Communicate(text, self.voice).stream():
                if chunk["type"] == "audio":
                    out += chunk["data"]
            return bytes(out)

        data = _run(go())
        if not data:
            raise ServiceError("tts", "Edge TTS returned no audio")
        return data


def get_tts():
    """ElevenLabs when a key is configured (or requested), otherwise the free Edge voice."""
    choice = os.getenv("TTS_PROVIDER", "").lower() or ("elevenlabs" if os.getenv("ELEVENLABS_API_KEY") else "edge")
    if choice == "elevenlabs":
        return ElevenLabsTTS(os.getenv("ELEVENLABS_API_KEY", ""), os.getenv("ELEVENLABS_VOICE", "21m00Tcm4TlvDq8ikWAM"), os.getenv("ELEVENLABS_MODEL", "eleven_flash_v2_5"))
    return EdgeTTS(os.getenv("EDGE_VOICE", "en-US-AriaNeural"))
