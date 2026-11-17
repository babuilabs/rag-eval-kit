"""Per-question scores and bootstrap confidence intervals."""
from __future__ import annotations

import random
from statistics import mean, median

from .corpus import evidence_overlap

HIT = 0.5  # a chunk "contains" an evidence quote if it holds >= half of the quote's 5-grams


def retrieval_scores(item: dict, retrieved: list[str]) -> dict:
    """Evidence recall within the retrieved list, and reciprocal rank of the first hit."""
    if not item["evidence"]:
        return {}
    found, first = [], None
    for ev in item["evidence"]:
        ranks = [r for r, text in enumerate(retrieved) if evidence_overlap(ev["quote"], text) >= HIT]
        found.append(bool(ranks))
        if ranks and (first is None or ranks[0] < first):
            first = ranks[0]
    return {"recall": sum(found) / len(found), "all_found": all(found),
            "rr": 0.0 if first is None else 1.0 / (first + 1)}


def question_score(item: dict, verdict: dict) -> float:
    """1 = the user got what they needed: correct answer, or an honest 'I don't know'."""
    if item["type"] == "unanswerable":
        return 1.0 if verdict.get("refused") else 0.0
    return {"correct": 1.0, "partial": 0.5}.get(verdict.get("correctness"), 0.0)


def bootstrap_ci(values: list[float], n: int = 2000, seed: int = 0) -> tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    means = sorted(mean(rng.choices(values, k=len(values))) for _ in range(n))
    return (means[int(0.025 * n)], means[int(0.975 * n) - 1])


def summarize(rows: list[dict]) -> dict:
    ans = [r for r in rows if r["type"] != "unanswerable"]
    una = [r for r in rows if r["type"] == "unanswerable"]

    def stat(vals):
        lo, hi = bootstrap_ci(vals)
        return {"value": round(mean(vals), 4) if vals else None, "ci95": [round(lo, 4), round(hi, 4)], "n": len(vals)}

    out = {
        "overall": stat([r["score"] for r in rows]),
        "answer_accuracy": stat([r["score"] for r in ans]),
        "retrieval_recall": stat([r["retrieval"]["recall"] for r in ans]),
        "retrieval_mrr": stat([r["retrieval"]["rr"] for r in ans]),
        "faithfulness": stat([1.0 if r["verdict"].get("faithful") else 0.0 for r in rows if not r["verdict"].get("refused")]),
        "false_refusal_rate": stat([1.0 if r["verdict"].get("refused") else 0.0 for r in ans]),
        "unanswerable_refusal": stat([r["score"] for r in una]),
        "latency_median_s": round(median(r["latency_s"] for r in rows), 3) if rows else None,
        "prompt_tokens_median": median(r.get("prompt_tokens", 0) for r in rows) if rows else None,
        "by_type": {},
    }
    for t in sorted({r["type"] for r in rows}):
        out["by_type"][t] = stat([r["score"] for r in rows if r["type"] == t])
    return out
