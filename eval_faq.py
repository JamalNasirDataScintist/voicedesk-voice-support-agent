"""Measure FAQ retrieval quality (what the model sees when it calls search_faq).

  python eval_faq.py

Settings were chosen on the 'tuning' questions only; 'held_out' questions were written afterwards.
The model reads every returned result, so top-3 recall is the number that matters for answer quality.
"""
from __future__ import annotations

import json
from pathlib import Path

from voicedesk.tools import ToolBox, load_embedder

ROOT = Path(__file__).parent


def evaluate(box: ToolBox, cases: list[dict], k: int = 3) -> tuple[float, float]:
    top1 = top_k = 0
    for case in cases:
        topics = [r["topic"] for r in box.search_faq(case["q"], top_k=k)["results"]]
        top1 += bool(topics) and topics[0] == case["topic"]
        top_k += case["topic"] in topics
    return top1 / len(cases), top_k / len(cases)


if __name__ == "__main__":
    data = json.loads((ROOT / "eval" / "faq_questions.json").read_text(encoding="utf-8"))
    embedder = load_embedder()
    modes = {"keyword (BM25)": ToolBox()}
    if embedder is not None:
        modes["hybrid (BM25 + embeddings)"] = ToolBox(embedder=embedder)
    else:
        print("sentence-transformers not installed: skipping the hybrid mode (pip install sentence-transformers)", end=chr(10) * 2)
    print(f"{'mode':<30}{'set':<10}{'top-1':>7}{'top-3':>7}")
    for mode, box in modes.items():
        for name in ("tuning", "held_out"):
            t1, t3 = evaluate(box, data[name])
            print(f"{mode:<30}{name:<10}{t1:>7.2f}{t3:>7.2f}")
        refused = sum(1 for q in data["unanswerable"] if not box.search_faq(q)["results"])
        print(f"{mode:<30}{'unanswerable correctly empty: ' + str(refused) + '/' + str(len(data['unanswerable']))}")
