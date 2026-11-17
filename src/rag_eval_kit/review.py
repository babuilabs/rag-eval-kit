"""Human review: is the eval set sound, and does the LLM judge agree with a person?

    python -m rag_eval_kit.review sheets                 # write review/*.csv to fill in
    python -m rag_eval_kit.review agreement              # score the filled-in sheets

Sheet 1 (questions): a reviewer marks whether each question is clear and its reference
answer is right. Sheet 2 (judgments): a reviewer grades assistant answers WITHOUT seeing the
judge's verdict; agreement with the judge is reported as accuracy and Cohen's kappa.
"""
from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

REVIEW = Path("review")
SEED = 7


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:  # BOM so Excel reads UTF-8
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def sheets(n_questions: int = 40, n_judgments: int = 50, runs=("v0_baseline", "v4_grounded_prompt")) -> None:
    rng = random.Random(SEED)
    full = {json.loads(l)["id"]: json.loads(l) for l in
            Path("results/evalset_build/kept_full.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    qs = rng.sample(sorted(full), min(n_questions, len(full)))
    _write(REVIEW / "1_questions.csv", [{
        "id": i, "type": full[i]["type"], "question": full[i]["question"],
        "reference_answer": full[i]["answer"] or "(should be: not in the handbook)",
        "source_text": "\n---\n".join(f"[{s['title']} > {s['head']}]\n{s['text']}" for s in full[i]["source"]),
        "question_ok (y/n)": "", "answer_ok (y/n)": "", "note": "",
    } for i in sorted(qs)])

    pool = []
    for run in runs:
        p = Path("results") / run / "answers.jsonl"
        if p.exists():
            pool += [dict(json.loads(l), run=run) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    if pool:
        picks = rng.sample(pool, min(n_judgments, len(pool)))
        rng.shuffle(picks)
        _write(REVIEW / "2_judgments.csv", [{
            "key": f"{r['run']}:{r['id']}", "question": r["question"],
            "reference_answer": r["reference"] or "(not in the handbook: a good answer says it doesn't know)",
            "assistant_answer": r["answer"],
            "your_grade (correct/partial/incorrect)": "", "note": "",
        } for r in picks])
    print(f"wrote {REVIEW}/1_questions.csv" + (f" and {REVIEW}/2_judgments.csv" if pool else ""))


def _kappa(a: list[str], b: list[str]) -> float:
    labels = sorted(set(a) | set(b))
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def agreement() -> None:
    out = {}
    q = [r for r in _read(REVIEW / "1_questions.csv") if r["question_ok (y/n)"].strip()]
    if q:
        ok = [r for r in q if r["question_ok (y/n)"].strip().lower().startswith("y")
              and not r["answer_ok (y/n)"].strip().lower().startswith("n")]
        out["evalset_items_reviewed"] = len(q)
        out["evalset_items_valid"] = len(ok)
        out["evalset_valid_rate"] = round(len(ok) / len(q), 3)
        out["evalset_flagged_ids"] = [r["id"] for r in q if r not in ok]
    p = REVIEW / "2_judgments.csv"
    if p.exists():
        rows = [r for r in _read(p) if r["your_grade (correct/partial/incorrect)"].strip()]
        human, judge = [], []
        for r in rows:
            run, qid = r["key"].split(":")
            ans = {json.loads(l)["id"]: json.loads(l) for l in
                   (Path("results") / run / "answers.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}[qid]
            h = r["your_grade (correct/partial/incorrect)"].strip().lower()
            if ans["type"] == "unanswerable":
                j = "correct" if ans["verdict"].get("refused") else "incorrect"
            else:
                j = ans["verdict"].get("correctness", "incorrect")
            human.append(h)
            judge.append(j)
        if human:
            out["judgments_reviewed"] = len(human)
            out["judge_human_exact_agreement"] = round(sum(a == b for a, b in zip(human, judge)) / len(human), 3)
            out["judge_human_kappa"] = round(_kappa(human, judge), 3)
            out["disagreements"] = [r["key"] for r, a, b in zip(rows, human, judge) if a != b]
    Path("results/human_review.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    cmd = sys.argv[1:2]
    if cmd == ["sheets"]:
        sheets()
    elif cmd == ["agreement"]:
        agreement()
    else:
        sys.exit(__doc__)
