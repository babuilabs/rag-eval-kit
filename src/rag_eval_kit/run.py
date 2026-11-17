"""Run one pipeline configuration over the evaluation set and score it.

    python -m rag_eval_kit.run configs/v0_baseline.yaml [--limit N]

Writes results/<name>/answers.jsonl (every answer, verdict and retrieved chunk id) and
results/<name>/metrics.json.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

from .chunking import chunk_corpus
from .corpus import load_corpus
from .judge import JUDGE_MODEL, judge
from .metrics import question_score, retrieval_scores, summarize
from .pipeline import GEN_MODEL, answer
from .retrieval import build_retriever


def load_evalset(path="data/evalset.jsonl") -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--generate-only", action="store_true",
                    help="only produce (and cache) answers; judge later. Useful under daily token caps")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    items = load_evalset()[: args.limit]

    docs = load_corpus("data/corpus")
    ch = cfg["chunking"]
    chunks = chunk_corpus(docs, ch["strategy"], **ch.get("params", {}))
    retr = build_retriever(chunks, cfg["retriever"]["kind"], cfg["retriever"].get("rerank", False))
    k = cfg["k"]
    print(f"{cfg['name']}: {len(chunks)} chunks, {len(items)} questions", flush=True)

    hits = {it["id"]: retr.search(it["question"], k) for it in items}

    def one(it):
        idx = [i for i, _ in hits[it["id"]]]
        texts = [chunks[i].text for i in idx]
        gen = answer(it["question"], texts, cfg["prompt"])
        if args.generate_only:
            return None
        verdict = judge(it, gen["answer"], "\n\n".join(texts))
        return {
            "id": it["id"], "type": it["type"], "question": it["question"], "reference": it["answer"],
            "answer": gen["answer"], "retrieved": [chunks[i].chunk_id for i in idx],
            "retrieval": retrieval_scores(it, texts), "verdict": verdict,
            "score": question_score(it, verdict), "latency_s": gen["latency_s"],
            "prompt_tokens": gen["usage"].get("prompt_tokens", 0),
        }

    rows = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for n, row in enumerate(pool.map(one, items), 1):
            rows.append(row)
            if n % 20 == 0:
                print(f"  {n}/{len(items)}", flush=True)
    if args.generate_only:
        print(f"{cfg['name']}: {len(items)} answers generated and cached", flush=True)
        return

    out = Path("results") / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "answers.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    from .llm import OPENROUTER_HOSTS, PROVIDER
    metrics = {"config": cfg, "generator": GEN_MODEL, "judge": JUDGE_MODEL, "n_chunks": len(chunks),
               "api": PROVIDER, "hosts": {m: OPENROUTER_HOSTS.get(m) for m in (GEN_MODEL, JUDGE_MODEL)} if PROVIDER == "openrouter" else None,
               **summarize(rows)}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({k: metrics[k] for k in ("overall", "answer_accuracy", "retrieval_recall",
                                              "faithfulness", "unanswerable_refusal")}, indent=1))


if __name__ == "__main__":
    main()
