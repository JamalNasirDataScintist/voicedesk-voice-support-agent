"""Offline tests: scripted fake LLM, fake STT and TTS. Nothing here touches the network."""
import base64
import json
from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from voicedesk import clients
from voicedesk.agent import EMPTY_REPLY, FALLBACK_REPLY, Agent, trim_history
from voicedesk.clients import ElevenLabsTTS, ServiceError
from voicedesk.guardrails import clean_user_text, redact, to_speech
from voicedesk.server import RateLimiter, Sessions, create_app
from voicedesk.tools import TOOL_SCHEMAS, ToolBox

SATURDAY_1030 = lambda: datetime(2026, 10, 3, 10, 30)  # a Saturday


def tool_call(name, args, call_id="call_1"):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args) if isinstance(args, dict) else args}}


class ScriptedLLM:
    def __init__(self, *messages):
        self.messages, self.calls = list(messages), []

    def __call__(self, messages, tools):
        self.calls.append(messages)
        return self.messages.pop(0) if len(self.messages) > 1 else self.messages[0]


# ------------------------------------------------------------------ tools

@pytest.fixture
def box():
    return ToolBox(clock=SATURDAY_1030)


def test_order_lookup_requires_matching_email(box):
    ok = box.lookup_order("A1001", "Maria@Example.com")
    assert ok["status"] == "shipped" and ok["tracking"].startswith("1Z") and ok["estimated_delivery"] == "2026-10-06"
    wrong_email = box.lookup_order("A1001", "someone@else.com")
    missing_order = box.lookup_order("A9999", "maria@example.com")
    assert wrong_email == missing_order and wrong_email["error"] == "not_found"  # callers cannot probe which orders exist
    assert "tracking" not in wrong_email and "items" not in wrong_email


def test_order_lookup_validates_format_and_normalises(box):
    assert box.lookup_order("12345", "x@y.com")["error"] == "invalid_order_number"
    assert box.lookup_order(" a 1002 ", "james@example.com")["status"] == "processing"
    assert box.lookup_order("A1002", "james@example.com")["can_be_cancelled"] is True
    delivered = box.lookup_order("A1003", "priya@example.com")
    assert delivered["status"] == "delivered" and "tracking" not in delivered


def test_faq_search_returns_candidates_including_the_right_policy(box):
    topics = [r["topic"] for r in box.search_faq("how long do I have to send something back for a refund")["results"]]
    assert "Returns" in topics and len(topics) <= 3
    returns = next(r for r in box.search_faq("returns and refunds")["results"] if r["topic"] == "Returns")
    assert "30 days" in returns["answer"]
    assert box.search_faq("do you sell motorbikes")["results"] == [] and "hint" in box.search_faq("zzzz qqqq")
    assert box.search_faq("")["results"] == []


class WordEmbedder:
    """Deterministic bag-of-words embedder over the FAQ vocabulary, enough to test the fusion and threshold logic."""

    def __init__(self, texts):
        words = sorted({w for t in texts for w in t.lower().replace(".", " ").split()})
        self.index = {w: i for i, w in enumerate(words)}

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        import numpy as np

        out = np.zeros((len(texts), len(self.index)), dtype="float32")
        for row, text in enumerate(texts):
            for w in text.lower().replace(".", " ").split():
                if w in self.index:
                    out[row, self.index[w]] += 1
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)


def test_hybrid_search_fuses_rankings_and_still_refuses_nonsense():
    plain = ToolBox()
    embedder = WordEmbedder([f"{t}. {b}" for t, b in plain.faq])
    hybrid = ToolBox(embedder=embedder)
    result = hybrid.search_faq("gift cards never expire", top_k=2)["results"]
    assert result[0]["topic"] == "Gift cards" and len(result) == 2
    assert hybrid.search_faq("zzzz qqqq xxxx")["results"] == []  # no keyword hit and similarity below the threshold


def test_store_hours_open_now_uses_the_injected_clock(box):
    today = box.get_store_hours()
    assert today["day"] == "saturday" and today["hours"] == "10:00-16:00" and today["open_now"] is True
    assert box.get_store_hours("Sunday") == {"day": "sunday", "hours": "closed"}
    assert box.get_store_hours("funday")["error"] == "unknown_day"
    late = ToolBox(clock=lambda: datetime(2026, 10, 5, 19, 0))  # Monday evening
    assert late.get_store_hours()["open_now"] is False


def test_callback_is_persisted_and_validated(box):
    first = box.request_callback("Ana", "+1 555 010 4477", "refund question", "tomorrow morning")
    second = box.request_callback("Ben", "555-010-9988", "broken item")
    assert first["ticket_id"] == "CB-0001" and second["ticket_id"] == "CB-0002"
    assert box.db.execute("SELECT name, phone FROM callbacks WHERE id = 1").fetchone() == ("Ana", "+1 555 010 4477")
    assert box.request_callback("Ana", "12", "x")["error"] == "invalid_phone"
    assert box.request_callback("", "555-010-9988", "x")["error"] == "missing_fields"


def test_dispatch_treats_model_arguments_as_untrusted(box):
    assert box.call("lookup_order", "not json")["error"] == "bad_arguments"
    assert box.call("lookup_order", '["a"]')["error"] == "bad_arguments"
    assert box.call("lookup_order", {"order_id": "A1001"})["error"] == "bad_arguments"  # missing email
    assert box.call("lookup_order", {"order_id": "A1001", "email": "maria@example.com", "admin": True})["error"] == "bad_arguments"
    for forbidden in ("db", "__init__", "orders", "call", "_load_faq", "os.system"):
        assert box.call(forbidden, {})["error"] == "unknown_tool"
    assert {s["function"]["name"] for s in TOOL_SCHEMAS} == {"lookup_order", "search_faq", "get_store_hours", "request_callback", "escalate_to_human"}


# ------------------------------------------------------------------ guardrails

def test_redact_masks_personal_data():
    text = "Mail maria@example.com or call +1 555 010 4477, card 4111111111111111"
    out = redact(text)
    assert "maria" not in out and "555" not in out and "4111" not in out and "[email]" in out and "[phone]" in out


def test_to_speech_cleans_and_truncates_on_sentence_boundary():
    assert to_speech("**Sure!** See https://x.io/a \U0001F600 now") == "Sure! See now"
    assert to_speech("Options:" + chr(10) + "- first" + chr(10) + "- second") == "Options: first second"
    long = "This is a sentence that goes on for a while. " * 20
    spoken = to_speech(long, limit=200)
    assert len(spoken) <= 200 and spoken.endswith(".")
    assert clean_user_text("  hello \n\n world  ") == "hello world" and len(clean_user_text("x" * 5000)) == 600


# ------------------------------------------------------------------ agent loop

def test_plain_reply_is_spoken_clean():
    agent = Agent(ScriptedLLM({"content": "**Hello!** How can I help? \U0001F60A"}), ToolBox())
    history = []
    result = agent.turn("hi there", history)
    assert result.reply == "Hello! How can I help?" and result.rounds == 1 and not result.escalated
    assert [m["role"] for m in history] == ["user", "assistant"]


def test_tool_round_trip_links_call_and_result(box):
    llm = ScriptedLLM(
        {"content": None, "tool_calls": [tool_call("lookup_order", {"order_id": "A1001", "email": "maria@example.com"}, "c1")]},
        {"content": "Your order has shipped with UPS and should arrive on October sixth."},
    )
    history = []
    result = Agent(llm, box).turn("Where is order A1001? My email is maria@example.com", history)
    assert result.rounds == 2 and result.tool_events[0]["name"] == "lookup_order" and result.tool_events[0]["result"]["status"] == "shipped"
    tool_msg = next(m for m in history if m["role"] == "tool")
    assert tool_msg["tool_call_id"] == "c1" and json.loads(tool_msg["content"])["tracking"].startswith("1Z")
    assert "shipped" in result.reply
    sent_to_model = llm.calls[1]
    assert sent_to_model[0]["role"] == "system" and any(m["role"] == "tool" for m in sent_to_model)


def test_bad_tool_arguments_are_returned_to_the_model_not_raised(box):
    llm = ScriptedLLM({"content": "", "tool_calls": [tool_call("lookup_order", "{broken")]}, {"content": "Could you repeat your order number?"})
    result = Agent(llm, box).turn("check my order", [])
    assert result.tool_events[0]["result"]["error"] == "bad_arguments" and result.reply.startswith("Could you repeat")


def test_runaway_tool_loop_is_capped_and_escalates(box):
    llm = ScriptedLLM({"content": "", "tool_calls": [tool_call("search_faq", {"query": "returns"})]})
    result = Agent(llm, box, max_tool_rounds=3).turn("returns?", [])
    assert result.reply == FALLBACK_REPLY and result.escalated and len(llm.calls) == 3
    assert box.escalations == ["agent exceeded the tool-call limit"]


def test_escalation_tool_flags_the_turn(box):
    llm = ScriptedLLM({"content": "", "tool_calls": [tool_call("escalate_to_human", {"reason": "caller is angry"})]}, {"content": "I'm connecting you to a colleague now."})
    result = Agent(llm, box).turn("this is ridiculous, get me a person", [])
    assert result.escalated and box.escalations == ["caller is angry"]


def test_empty_utterance_never_reaches_the_model():
    llm = ScriptedLLM({"content": "should not be used"})
    assert Agent(llm, ToolBox()).turn("   ", []).reply == EMPTY_REPLY and llm.calls == []


def test_history_trim_never_splits_a_tool_call_from_its_result():
    history = []
    for i in range(14):
        history += [{"role": "user", "content": f"q{i}"},
                    {"role": "assistant", "content": "", "tool_calls": [tool_call("search_faq", {"query": "x"}, f"id{i}")]},
                    {"role": "tool", "tool_call_id": f"id{i}", "content": "{}"},
                    {"role": "assistant", "content": f"a{i}"}]
    trimmed = trim_history(history, 10)
    assert trimmed[0]["role"] == "user" and trimmed[0]["content"] == "q4" and sum(m["role"] == "user" for m in trimmed) == 10
    assert all(m["role"] != "tool" or any(c["id"] == m["tool_call_id"] for p in trimmed for c in p.get("tool_calls", [])) for m in trimmed)


# ------------------------------------------------------------------ clients

class Resp:
    def __init__(self, status, content=b"", text=""):
        self.status_code, self.content, self.text = status, content, text


def test_elevenlabs_retries_then_fails_clearly(monkeypatch):
    statuses = iter([429, 503, 200])
    monkeypatch.setattr(clients.requests, "post", lambda *a, **k: Resp(next(statuses), b"MP3", "busy"))
    sleeps = []
    assert ElevenLabsTTS("k", sleep=sleeps.append).synthesize("hi") == b"MP3" and sleeps == [1, 2]
    monkeypatch.setattr(clients.requests, "post", lambda *a, **k: Resp(401, b"", "no"))
    with pytest.raises(ServiceError, match="rejected the API key"):
        ElevenLabsTTS("k").synthesize("hi")
    monkeypatch.setattr(clients.requests, "post", lambda *a, **k: Resp(400, b"", "quota_exceeded"))
    with pytest.raises(ServiceError, match="quota"):
        ElevenLabsTTS("k").synthesize("hi")
    with pytest.raises(ServiceError):
        ElevenLabsTTS("")


def test_transcribe_request_shape_and_limits(monkeypatch):
    seen = {}
    monkeypatch.setenv("GROQ_API_KEY", "gk")

    def fake_post(url, headers=None, files=None, data=None, timeout=None):
        seen.update(url=url, headers=headers, files=files, data=data)
        return type("R", (), {"status_code": 200, "json": lambda self: {"text": " hello world "}})()

    monkeypatch.setattr(clients.requests, "post", fake_post)
    assert clients.transcribe(b"audio-bytes", "a.webm", language="en") == "hello world"
    assert seen["url"].endswith("/audio/transcriptions") and seen["headers"]["Authorization"] == "Bearer gk" and seen["data"]["language"] == "en"
    with pytest.raises(ServiceError, match="Empty"):
        clients.transcribe(b"")
    with pytest.raises(ServiceError, match="too large"):
        clients.transcribe(b"x" * (clients.MAX_AUDIO_BYTES + 1))


def test_missing_groq_key_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ServiceError, match="GROQ_API_KEY"):
        clients.chat_with_tools([], [])


# ------------------------------------------------------------------ API

class FakeTTS:
    name = "fake"

    def synthesize(self, text):
        return b"FAKE-MP3:" + text.encode()


def make_client(llm=None, stt=None, tts=None, rate_limit=1000):
    llm = llm or ScriptedLLM({"content": "Happy to help."})
    app = create_app(llm=llm, stt=stt or (lambda audio, name: "where is my order"), tts=tts or FakeTTS(), toolbox=ToolBox(), rate_limit=rate_limit)
    return TestClient(app), llm


SID = "session-abcdef"


def test_chat_returns_reply_audio_and_metrics():
    client, _ = make_client()
    body = client.post("/api/chat", json={"session_id": SID, "text": "hello"}).json()
    assert body["reply"] == "Happy to help." and body["tts_error"] is None
    assert base64.b64decode(body["audio_base64"]) == b"FAKE-MP3:Happy to help."
    assert set(body["metrics_ms"]) == {"stt", "llm", "tts", "total"}


def test_session_memory_persists_across_requests():
    client, llm = make_client()
    client.post("/api/chat", json={"session_id": SID, "text": "my name is Dana"})
    client.post("/api/chat", json={"session_id": SID, "text": "what did I just say?"})
    second_prompt = llm.calls[1]
    assert any(m.get("content") == "my name is Dana" for m in second_prompt)
    other = client.post("/api/chat", json={"session_id": "other-session-1", "text": "hi"})
    assert other.status_code == 200 and not any(m.get("content") == "my name is Dana" for m in llm.calls[2])  # sessions are isolated


def test_voice_endpoint_transcribes_then_answers():
    client, llm = make_client()
    res = client.post("/api/voice", files={"audio": ("clip.webm", b"\x1aE\xdf\xa3fake", "audio/webm")}, data={"session_id": SID})
    body = res.json()
    assert res.status_code == 200 and body["transcript"] == "where is my order" and body["metrics_ms"]["stt"] >= 0
    assert any(m.get("content") == "where is my order" for m in llm.calls[0])


def test_tts_failure_degrades_to_text_only():
    class Broken:
        def synthesize(self, text):
            raise ServiceError("tts", "ElevenLabs quota or plan limit reached")

    client, _ = make_client(tts=Broken())
    res = client.post("/api/chat", json={"session_id": SID, "text": "hello"})
    body = res.json()
    assert res.status_code == 200 and body["reply"] and body["audio_base64"] is None and "quota" in body["tts_error"]


def test_stt_failure_is_a_503_naming_the_service():
    def broken_stt(audio, name):
        raise ServiceError("stt", "Groq transcription returned 500")

    client, _ = make_client(stt=broken_stt)
    res = client.post("/api/voice", files={"audio": ("c.webm", b"data")}, data={"session_id": SID})
    assert res.status_code == 503 and res.json()["service"] == "stt"


def test_input_validation_and_abuse_protection():
    client, _ = make_client(rate_limit=3)
    assert client.post("/api/chat", json={"session_id": "../../etc", "text": "hi"}).status_code == 400
    assert client.post("/api/chat", json={"session_id": SID, "text": ""}).status_code == 422
    big = client.post("/api/voice", files={"audio": ("c.webm", b"x" * (clients.MAX_AUDIO_BYTES + 10))}, data={"session_id": SID})
    assert big.status_code in (413,)
    codes = [client.post("/api/chat", json={"session_id": SID, "text": "hi"}).status_code for _ in range(5)]
    assert 429 in codes


def test_health_and_index():
    client, _ = make_client()
    health = client.get("/api/health").json()
    assert health["status"] == "ok" and health["tts_provider"] == "fake"
    assert client.get("/").status_code == 200 and "VoiceDesk" in client.get("/").text


def test_rate_limiter_window_and_sessions_expiry_and_cap():
    t = [0.0]
    limiter = RateLimiter(2, clock=lambda: t[0])
    assert limiter.allow("a") and limiter.allow("a") and not limiter.allow("a") and limiter.allow("b")
    t[0] = 61
    assert limiter.allow("a")
    s = Sessions(ttl=10, max_sessions=2, clock=lambda: t[0])
    s.get("one").append(1)
    t[0] = 65
    s.get("two"); s.get("three")
    assert "one" not in s.items and len(s.items) == 2  # expired, and the cap evicts the oldest
