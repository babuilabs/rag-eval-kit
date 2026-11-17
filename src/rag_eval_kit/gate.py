"""Safe gate: check each answer against its passages before showing it, and hold back the
ones that are probably wrong (in production they would go to a person instead).

    python -m rag_eval_kit.gate configs/v4_grounded_prompt.yaml [--alpha 0.03 0.05 0.10]

Writes results/v5_safe_gate/<source run>/scores.jsonl and metrics.json.

1. A verifier from a third model family (neither the generator nor the judge) reads the
   question, the retrieved passages and the answer, and returns P(every claim is supported).
2. The threshold is set with conformal risk control (Angelopoulos et al., 2022) on a
   calibration split, so that the expected rate of wrong answers shown to users is at most
   alpha. Wrong = an incorrect answer (partial counts half), or any answer to a question
   the handbook cannot answer. Labels come from the judge.
3. The guarantee holds in expectation over calibration sets drawn from the same
   distribution. We report the realised risk and coverage over many random splits.

Any verifier that maps (question, passages, answer) to a probability can be swapped in.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean

import yaml

from .chunking import chunk_corpus
from .corpus import load_corpus
from .llm import chat
from .pipeline import format_context

VERIFIER_MODEL = "google/gemma-3-27b-it"

VERIFY = """Passages:
{context}

Question: {question}

Answer:
{answer}

Is every factual claim in the answer directly supported by the passages? Reply with one word: Yes or No."""


class LLMVerifier:
    """P(supported) from the log-probabilities of the first answer token."""

    def __init__(self, model: str = VERIFIER_MODEL):
        self.model = model

    def support_prob(self, question: str, passages: list[str], answer: str) -> float:
        prompt = VERIFY.format(context=format_context(passages, True), question=question, answer=answer)
        out = chat(self.model, [{"role": "user", "content": prompt}], max_tokens=1,
                   extra={"logprobs": True, "top_logprobs": 10})
        top = (out.get("logprobs") or {}).get("content", [{}])[0].get("top_logprobs", [])
        yes = sum(math.exp(t["logprob"]) for t in top if t["token"].strip().lower() == "yes")
        no = sum(math.exp(t["logprob"]) for t in top if t["token"].strip().lower() == "no")
        if yes + no == 0:
            raise RuntimeError(f"{self.model}: no Yes/No log-probabilities returned")
        return yes / (yes + no)


def loss(row: dict) -> float:
    """Cost of showing this answer: 1 if wrong, 0.5 if partial, 0 if right."""
    if row["type"] == "unanswerable":
        return 1.0
    return {"correct": 0.0, "partial": 0.5}.get(row["verdict"].get("correctness"), 1.0)


def risk(rows: list[dict], tau: float) -> float:
    """Share of all questions where a wrong answer is shown at threshold tau."""
    return mean(r["loss"] * (r["shown"] and r["p_supported"] >= tau) for r in rows)


def coverage(rows: list[dict], tau: float) -> float:
    return mean(r["shown"] and r["p_supported"] >= tau for r in rows)


def crc_threshold(cal: list[dict], alpha: float) -> float:
    """Smallest threshold whose adjusted calibration risk is <= alpha (losses bounded by 1)."""
    n = len(cal)
    for tau in sorted({0.0} | {r["p_supported"] for r in cal} | {1.0 + 1e-9}):
        if (n * risk(cal, tau) + 1) / (n + 1) <= alpha:
            return tau
    return 1.0 + 1e-9


def auroc(pos: list[float], neg: list[float]) -> float | None:
    """P(score of a right answer > score of a wrong one)."""
    if not pos or not neg:
        return None
    return mean((p > q) + 0.5 * (p == q) for p in pos for q in neg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--alpha", type=float, nargs="+", default=[0.03, 0.05, 0.10])
    ap.add_argument("--splits", type=int, default=1000)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    src = Path("results") / cfg["name"]
    rows = [json.loads(l) for l in (src / "answers.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

    ch = cfg["chunking"]
    text = {c.chunk_id: c.text for c in chunk_corpus(load_corpus("data/corpus"), ch["strategy"], **ch.get("params", {}))}
    verifier = LLMVerifier()

    def score(r):
        r = dict(r, loss=loss(r), shown=not r["verdict"].get("refused"))
        # A refusal is never shown as an answer, so it needs no check.
        r["p_supported"] = verifier.support_prob(r["question"], [text[c] for c in r["retrieved"]], r["answer"]) if r["shown"] else 0.0
        return r

    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(score, rows))

    out = Path("results/v5_safe_gate") / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "scores.jsonl").write_text("".join(json.dumps({k: r[k] for k in ("id", "type", "loss", "shown", "p_supported")}) + "\n"
                                              for r in rows), encoding="utf-8")

    shown = [r for r in rows if r["shown"]]
    report = {
        "source_run": cfg["name"], "verifier": verifier.model, "n": len(rows), "splits": args.splits,
        "no_gate": {"wrong_shown": round(risk(rows, 0.0), 4), "answered": round(coverage(rows, 0.0), 4)},
        "verifier_auroc": auroc([r["p_supported"] for r in shown if r["loss"] == 0],
                                [r["p_supported"] for r in shown if r["loss"] > 0]),
        "gate": {},
    }
    rng = random.Random(0)
    for alpha in args.alpha:
        risks, covs, right = [], [], []
        for _ in range(args.splits):
            idx = list(range(len(rows)))
            rng.shuffle(idx)
            half = len(idx) // 2
            cal, test = [rows[i] for i in idx[:half]], [rows[i] for i in idx[half:]]
            tau = crc_threshold(cal, alpha)
            risks.append(risk(test, tau))
            covs.append(coverage(test, tau))
            right.append(mean(r["shown"] and r["p_supported"] >= tau and r["loss"] == 0 for r in test))
        risks.sort()
        report["gate"][str(alpha)] = {
            "wrong_shown_mean": round(mean(risks), 4),
            "wrong_shown_p90": round(risks[int(0.9 * len(risks))], 4),
            "answered_mean": round(mean(covs), 4),
            "answered_right_mean": round(mean(right), 4),
            "calibration_n": len(rows) // 2,
        }
    curve = []
    for tau in sorted({r["p_supported"] for r in shown}):
        curve.append({"tau": round(tau, 6), "answered": round(coverage(rows, tau), 4), "wrong_shown": round(risk(rows, tau), 4)})
    report["risk_coverage_curve"] = curve
    (out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "risk_coverage_curve"}, indent=1))


if __name__ == "__main__":
    main()
