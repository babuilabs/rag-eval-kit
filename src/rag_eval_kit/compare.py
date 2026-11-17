"""Compare finished runs side by side.

    python -m rag_eval_kit.compare            # prints a markdown table, writes results/COMPARISON.md
"""
from __future__ import annotations

import json
from pathlib import Path

ORDER = ["v0_baseline", "v1_heading_chunks", "v2_hybrid", "v3_rerank", "v4_grounded_prompt"]
COLS = [("overall", "Overall"), ("answer_accuracy", "Answer accuracy"),
        ("retrieval_recall", "Evidence recall@k"), ("faithfulness", "Faithfulness"),
        ("unanswerable_refusal", "Says \"don't know\" when it should"),
        ("false_refusal_rate", "Refuses answerable")]


def pct(stat: dict) -> str:
    if not stat or stat.get("value") is None:
        return "–"
    lo, hi = stat["ci95"]
    return f"{stat['value'] * 100:.0f}% <sub>({lo * 100:.0f}–{hi * 100:.0f})</sub>"


def main() -> None:
    runs = [json.loads((Path("results") / r / "metrics.json").read_text())
            for r in ORDER if (Path("results") / r / "metrics.json").exists()]
    lines = ["| Version | Change | " + " | ".join(c[1] for c in COLS) + " |",
             "|---|---|" + "---|" * len(COLS)]
    for m in runs:
        cfg = m["config"]
        lines.append(f"| `{cfg['name']}` | {cfg['description']} | " + " | ".join(pct(m.get(k)) for k, _ in COLS) + " |")
    if runs:
        n = runs[0]["overall"]["n"]
        lines.append(f"\n{n} questions. 95% bootstrap confidence intervals in brackets. "
                     f"Generator `{runs[0]['generator']}`, judge `{runs[0]['judge']}`.")
    text = "\n".join(lines)
    Path("results/COMPARISON.md").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
