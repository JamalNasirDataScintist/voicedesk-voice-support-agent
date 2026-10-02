"""Business tools the agent can call. Each tool validates its own input and never trusts the model.

Security note: ``lookup_order`` requires the e-mail on the order. Without that check a caller could read
other customers' orders just by guessing order numbers.
"""
from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

DATA = Path(__file__).parent.parent / "data"
DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_TOKEN = re.compile(r"[a-z0-9]+")
ORDER_ID = re.compile(r"^[A-Z]\d{4}$")
PHONE = re.compile(r"^\+?[0-9][0-9 ()\-]{6,18}[0-9]$")

TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "lookup_order",
        "description": "Look up an order's status. The customer must give BOTH the order number (letter plus 4 digits, e.g. A1001) and the e-mail used on the order.",
        "parameters": {"type": "object", "properties": {
            "order_id": {"type": "string", "description": "Order number such as A1001"},
            "email": {"type": "string", "description": "E-mail address on the order"}},
            "required": ["order_id", "email"]}}},
    {"type": "function", "function": {
        "name": "search_faq",
        "description": "Search company policies: returns, delivery, warranty, gift cards, payments, order changes. Use before answering any policy question.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "get_store_hours",
        "description": "Opening hours for a given weekday, or for today when no day is given.",
        "parameters": {"type": "object", "properties": {"day": {"type": "string", "description": "Weekday name, optional"}}}}},
    {"type": "function", "function": {
        "name": "request_callback",
        "description": "Create a callback request for a human agent. Needs the caller's name, a phone number and a short reason.",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string"}, "phone": {"type": "string"},
            "reason": {"type": "string"}, "preferred_time": {"type": "string", "description": "Optional, free text"}},
            "required": ["name", "phone", "reason"]}}},
    {"type": "function", "function": {
        "name": "escalate_to_human",
        "description": "Hand the conversation to a human immediately. Use when the caller is upset, asks for a person, or the issue is outside what you can do.",
        "parameters": {"type": "object", "properties": {"reason": {"type": "string"}}, "required": ["reason"]}}},
]


STOPWORDS = frozenset(
    "a an the and or but if of to in on at for from with about as is are was were be been do does did can could would should will "
    "i me my we our you your it its this that these those there how what when where which who why long much many some any have has had get got "
    "please tell want need like just also so then than not no yes".split()
)
# Query-side expansion for everyday wording that differs from the policy text ("send it back" -> returns).
SYNONYMS = {
    "back": ["return"], "refund": ["return"], "arrive": ["deliveri"], "deliver": ["deliveri"], "ship": ["deliveri"], "parcel": ["deliveri"],
    "order": ["order"], "broken": ["damag"], "faulty": ["damag"], "defect": ["damag", "warranti"], "guarante": ["warranti"], "pay": ["payment"],
    "wallet": ["payment"], "swap": ["return"], "expir": ["gift"],
}


def _stem(word: str) -> str:
    """Very light stemmer: enough to match 'returns/returned', 'cancelling/cancelled', 'delivery/deliveries'."""
    for suffix in ("ies", "ing", "ed", "es", "s", "y"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)] + ("i" if suffix in ("ies", "y") else "")
            break
    return re.sub(r"([^aeiou])\1$", r"\1", word) if len(word) > 3 else word


def _tokens(text: str) -> list[str]:
    return [_stem(w) for w in _TOKEN.findall(text.lower()) if w not in STOPWORDS]


def _expand(tokens: list[str]) -> list[str]:
    extra = [t for token in tokens for t in SYNONYMS.get(token, [])]
    return tokens + extra


MIN_SIMILARITY = 0.22  # below this, with no keyword hit either, the question is treated as unanswerable
RRF_K = 60


def load_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Return a sentence-transformers model, or None when the optional dependency is not installed."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None
    return SentenceTransformer(model_name)


class ToolBox:
    def __init__(self, data_dir: Path = DATA, db_path: str | Path = ":memory:", clock=datetime.now, title_weight: int = 2, embedder=None):
        shop = json.loads((data_dir / "shop.json").read_text(encoding="utf-8"))
        self.orders: dict = shop["orders"]
        self.hours: dict = shop["hours"]
        self.clock = clock
        self.faq = self._load_faq((data_dir / "faq.md").read_text(encoding="utf-8"))
        # The title is repeated ``title_weight`` times so a topic's name outweighs incidental words in other sections.
        self._bm25 = BM25Okapi([_tokens(" ".join([title] * title_weight) + " " + body) for title, body in self.faq])
        # Optional semantic search: with an embedder the FAQ lookup fuses keyword and embedding rankings.
        self._embedder = embedder
        self._doc_vecs = None
        if embedder is not None:
            self._doc_vecs = self._embed([f"{t}. {b}" for t, b in self.faq])
        self.db = sqlite3.connect(str(db_path), check_same_thread=False)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS callbacks (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, phone TEXT, "
            "reason TEXT, preferred_time TEXT, created_at TEXT)"
        )
        self.escalations: list[str] = []

    @staticmethod
    def _load_faq(markdown: str) -> list[tuple[str, str]]:
        sections, title, body = [], None, []
        for line in markdown.splitlines():
            if line.startswith("# "):
                if title:
                    sections.append((title, " ".join(body).strip()))
                title, body = line[2:].strip(), []
            elif line.strip():
                body.append(line.strip())
        if title:
            sections.append((title, " ".join(body).strip()))
        return sections

    # ---- tools -------------------------------------------------------------------------------------------

    def lookup_order(self, order_id: str, email: str) -> dict:
        order_id = (order_id or "").strip().upper().replace(" ", "")
        if not ORDER_ID.match(order_id):
            return {"error": "invalid_order_number", "hint": "Order numbers are one letter followed by four digits, for example A1001."}
        order = self.orders.get(order_id)
        # Same answer for 'no such order' and 'wrong e-mail' so callers cannot probe which orders exist.
        if not order or order["email"].lower() != (email or "").strip().lower():
            return {"error": "not_found", "hint": "No order matches that order number and e-mail. Ask the caller to check both."}
        result = {"order_id": order_id, "status": order["status"], "items": order["items"], "total": order["total"]}
        if order["status"] == "shipped":
            result.update(carrier=order["carrier"], tracking=order["tracking"], estimated_delivery=order["eta"])
        elif order["status"] == "processing":
            result.update(estimated_ship_date=order["eta"], can_be_cancelled=True)
        return result

    def _embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self._embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False), dtype="float32")

    def search_faq(self, query: str, top_k: int = 3) -> dict:
        """Return up to ``top_k`` candidate policies. The model reads them all, so recall matters more than rank."""
        tokens = _expand(_tokens(query or ""))
        if not tokens and self._embedder is None:
            return {"results": []}
        keyword = self._bm25.get_scores(tokens) if tokens else np.zeros(len(self.faq))
        keyword_rank = [i for i in np.argsort(-keyword) if keyword[i] > 0]
        if self._embedder is None:
            order = keyword_rank[:top_k]
        else:
            sims = self._doc_vecs @ self._embed([query or ""])[0]
            if sims.max() < MIN_SIMILARITY and not keyword_rank:
                order = []
            else:
                dense_rank = list(np.argsort(-sims))
                fused: dict[int, float] = {}
                for ranking in (keyword_rank, dense_rank):
                    for position, i in enumerate(ranking, start=1):
                        fused[int(i)] = fused.get(int(i), 0.0) + 1.0 / (RRF_K + position)
                order = sorted(fused, key=fused.get, reverse=True)[:top_k]
        results = [{"topic": self.faq[int(i)][0], "answer": self.faq[int(i)][1]} for i in order]
        return {"results": results} if results else {"results": [], "hint": "No matching policy found. Offer a callback instead of guessing."}

    def get_store_hours(self, day: str | None = None) -> dict:
        now = self.clock()
        name = (day or DAYS[now.weekday()]).strip().lower()
        if name not in self.hours:
            return {"error": "unknown_day", "hint": f"Use one of: {', '.join(DAYS)}"}
        hours = self.hours[name]
        result = {"day": name, "hours": hours}
        if day is None and hours != "closed":
            start, end = (datetime.strptime(t, "%H:%M").time() for t in hours.split("-"))
            result["open_now"] = start <= now.time() < end
        return result

    def request_callback(self, name: str, phone: str, reason: str, preferred_time: str = "") -> dict:
        if not (name or "").strip() or not (reason or "").strip():
            return {"error": "missing_fields", "hint": "Name and reason are required."}
        if not PHONE.match((phone or "").strip()):
            return {"error": "invalid_phone", "hint": "Ask the caller to repeat the phone number including the area code."}
        cur = self.db.execute(
            "INSERT INTO callbacks (name, phone, reason, preferred_time, created_at) VALUES (?, ?, ?, ?, ?)",
            (name.strip()[:80], phone.strip(), reason.strip()[:300], (preferred_time or "")[:80], self.clock().isoformat(timespec="seconds")),
        )
        self.db.commit()
        return {"ticket_id": f"CB-{cur.lastrowid:04d}", "status": "created"}

    def escalate_to_human(self, reason: str) -> dict:
        self.escalations.append((reason or "unspecified").strip()[:200])
        return {"status": "handoff_requested"}

    # ---- dispatch ----------------------------------------------------------------------------------------

    def call(self, name: str, arguments: str | dict) -> dict:
        """Run a tool by name with model-supplied (untrusted) arguments. Always returns a JSON-serialisable dict."""
        try:
            args = json.loads(arguments) if isinstance(arguments, str) else dict(arguments or {})
            if not isinstance(args, dict):
                raise ValueError("arguments must be an object")
        except (json.JSONDecodeError, ValueError, TypeError):
            return {"error": "bad_arguments", "hint": "Arguments were not valid JSON."}
        fn = getattr(self, name, None) if name in {s["function"]["name"] for s in TOOL_SCHEMAS} else None
        if fn is None:
            return {"error": "unknown_tool", "hint": f"No tool called '{name}'."}
        try:
            return fn(**args)
        except TypeError as exc:
            return {"error": "bad_arguments", "hint": str(exc)[:160]}
