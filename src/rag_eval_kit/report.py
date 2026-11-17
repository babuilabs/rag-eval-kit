"""Build the HTML report (docs/index.html) from the finished runs.

    python -m rag_eval_kit.report

Reads only results/, so it needs no API key and no network.
"""
from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np

from .compare import ORDER, stat
from .gate import loss

RESULTS = Path("results")
# Kept in the eval set but flagged in the audit as ambiguous or awkward; never used as showcase examples.
WEAK = {"q018", "q022", "q025", "q056", "q068", "q070", "q078", "q081", "q088", "q090", "q093", "q096"}
OUT = Path("docs/index.html")
JS = Path(__file__).with_name("report.js")

SHORT = {
    "v0_baseline": "Tutorial baseline",
    "v1_heading_chunks": "+ heading-aware chunks",
    "v2_hybrid": "+ hybrid search",
    "v3_rerank": "+ reranker",
    "v4_grounded_prompt": "+ grounded prompt",
    "v6_more_context": "+ 6 passages",
    "v7_decompose": "+ split comparisons",
}


def _rows(run: str) -> list[dict]:
    return [json.loads(l) for l in (RESULTS / run / "answers.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]


def wrong_shown(rows: list[dict]) -> float:
    return sum(loss(r) * (not r["verdict"].get("refused")) for r in rows) / len(rows)


def _pct(x: float | None, digits: int = 0) -> str:
    return "–" if x is None else f"{x * 100:.{digits}f}%"


def _ci(s: dict | None) -> str:
    if not s or s.get("value") is None:
        return "–"
    lo, hi = s["ci95"]
    return f'{s["value"] * 100:.0f}%<span class="ci">{lo * 100:.0f}–{hi * 100:.0f}</span>'


def _table(runs: list[tuple[str, dict, list[dict]]]) -> str:
    head = ("<tr><th>Version</th><th>Change</th><th>Overall</th><th>Answer accuracy</th><th>Evidence recall</th>"
            "<th>Faithful</th><th>Says “don’t know” when it should</th><th>Refuses answerable</th>"
            "<th>Multi-hop</th><th>Wrong shown</th></tr>")
    body = "".join(
        f'<tr><td><code>{n}</code></td><td>{html.escape(m["config"]["description"])}</td>'
        f'<td>{_ci(m["overall"])}</td><td>{_ci(m["answer_accuracy"])}</td><td>{_ci(m["retrieval_recall"])}</td>'
        f'<td>{_ci(m["faithfulness"])}</td><td>{_ci(m["unanswerable_refusal"])}</td>'
        f'<td>{_ci(m["false_refusal_rate"])}</td><td>{_ci(stat(m, "multi_hop"))}</td>'
        f'<td>{_pct(wrong_shown(rows), 1)}</td></tr>'
        for n, m, rows in runs)
    return f'<div class="table-wrap"><table>{head}{body}</table></div>'


def _gate() -> str:
    cards = []
    for p in sorted((RESULTS / "v5_safe_gate").glob("*/metrics.json"), key=lambda p: ORDER.index(p.parent.name)
                    if p.parent.name in ORDER else 99):
        g = json.loads(p.read_text(encoding="utf-8"))
        rows = "".join(
            f'<tr><td>α = {a}</td><td>{_pct(v["wrong_shown_mean"], 1)}</td><td>{_pct(v["wrong_shown_p90"], 1)}</td>'
            f'<td>{_pct(v["answered_mean"])}</td></tr>' for a, v in g["gate"].items())
        cards.append(
            f'<div class="gate"><h3>Gate on <code>{g["source_run"]}</code></h3>'
            f'<p class="note">Verifier AUROC {g["verifier_auroc"]:.2f} · without the gate: '
            f'{_pct(g["no_gate"]["wrong_shown"], 1)} wrong shown, {_pct(g["no_gate"]["answered"])} answered · '
            f'{g["splits"]} random splits, {next(iter(g["gate"].values()))["calibration_n"]} calibration questions each</p>'
            f'<table><tr><th>Target</th><th>Wrong shown (mean)</th><th>Wrong shown (90th pct)</th><th>Answered</th></tr>{rows}</table></div>')
    return "".join(cards)


ALPHAS = [round(0.01 + 0.005 * i, 3) for i in range(29)]  # 1% … 15%


def gate_grid(source: str, splits: int = 1000, seed: int = 0) -> dict | None:
    """Held-out gate results for a grid of targets, the same procedure as `gate.py`:
    half the questions calibrate the threshold (conformal risk control), the other half test it."""
    path = RESULTS / "v5_safe_gate" / source / "scores.jsonl"
    if not path.exists():
        return None
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    # A refusal is never shown, so give it a score below every threshold.
    p = np.array([r["p_supported"] if r["shown"] else -1.0 for r in rows])
    lo = np.array([r["loss"] for r in rows])
    n, half = len(rows), len(rows) // 2
    rng = np.random.default_rng(seed)
    perms = [rng.permutation(n) for _ in range(splits)]
    out = {}
    for alpha in ALPHAS:
        wrong, answered, right = [], [], []
        held_n, test_n = np.zeros(n), np.zeros(n)  # how often each question was held back when it was in the test half
        for idx in perms:
            cal, test = idx[:half], idx[half:]
            taus = np.unique(np.concatenate([[0.0, 1.0 + 1e-9], p[cal][p[cal] >= 0]]))
            cal_risk = (lo[cal][:, None] * (p[cal][:, None] >= taus[None, :])).sum(0) / half
            ok = (half * cal_risk + 1) / (half + 1) <= alpha
            tau = taus[ok.argmax()] if ok.any() else 1.0 + 1e-9
            shown = p[test] >= tau
            test_n[test] += 1
            held_n[test] += (p[test] >= 0) & ~shown
            wrong.append((lo[test] * shown).mean())
            answered.append(shown.mean())
            right.append((shown & (lo[test] == 0)).mean())
        wrong = np.sort(wrong)
        out[str(alpha)] = {"wrong": round(float(np.mean(wrong)), 4), "wrong_p90": round(float(wrong[int(0.9 * splits)]), 4),
                           "answered": round(float(np.mean(answered)), 4), "right": round(float(np.mean(right)), 4),
                           "held_freq": [round(float(h / max(t, 1)), 3) for h, t in zip(held_n, test_n)]}
    shown_all = p >= 0
    return {"source": source, "n": n, "calibration_n": half, "splits": splits,
            "ids": [r["id"] for r in rows],
            "no_gate": {"wrong": round(float((lo * shown_all).mean()), 4), "answered": round(float(shown_all.mean()), 4),
                        "right": round(float((shown_all & (lo == 0)).mean()), 4)},
            "grid": out}


def explorer_data(runs: list[tuple[str, dict, list[dict]]]) -> dict:
    """Every question with every version's answer and verdict, for the in-page explorer."""
    qs, answers = {}, {}
    for name, _, rows in runs:
        answers[name] = {}
        for r in rows:
            qs.setdefault(r["id"], {"id": r["id"], "type": r["type"], "q": r["question"], "ref": r["reference"],
                                    "weak": r["id"] in WEAK})
            v = r["verdict"]
            answers[name][r["id"]] = {"a": r["answer"], "s": r["score"], "c": v.get("correctness"),
                                      "x": bool(v.get("refused")), "why": v.get("reason", ""),
                                      "rec": (r.get("retrieval") or {}).get("recall")}
    return {"versions": [{"id": n, "label": f'{n.split("_")[0]} {SHORT.get(n, n)}'} for n, _, _ in runs],
            "questions": list(qs.values()), "answers": answers}


def _json_script(element_id: str, data) -> str:
    text = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return f'<script type="application/json" id="{element_id}">{text}</script>'


STEPS = {
    "v0_baseline": "Tutorial baseline: 1000-character chunks, top-3 vector search, plain prompt.",
    "v1_heading_chunks": "Split the documents on their headings instead of every 1000 characters.",
    "v2_hybrid": "Add keyword search (BM25). On this corpus it made things slightly worse.",
    "v3_rerank": "Re-rank the candidate passages with a cross-encoder.",
    "v4_grounded_prompt": "Prompt it to cite passages and say “I don’t know” when the answer isn’t there.",
    "v6_more_context": "Give it six passages instead of three.",
    "v7_decompose": "Search each side of a comparison question separately.",
}
CARD_IDS = ("q106", "q076", "q066")  # one unanswerable, one comparison, one fact; chosen from the explorer


def _examples(before: list[dict], after: list[dict]) -> str:
    """Three short before/after cards. Full answers sit behind a toggle."""
    a, b = {r["id"]: r for r in before}, {r["id"]: r for r in after}
    ids = [i for i in CARD_IDS if i in a and i in b and a[i]["score"] < 1 <= b[i]["score"] and i not in WEAK]
    for r in before:  # fall back to the first fixed question of another type
        if len(ids) >= 3:
            break
        x = b.get(r["id"])
        if x and r["id"] not in ids and r["id"] not in WEAK and r["score"] == 0 and x["score"] == 1 \
                and r["type"] not in {a[i]["type"] for i in ids}:
            ids.append(r["id"])
    clean = lambda s: html.escape(s.replace("**", ""))
    cards = []
    for i in ids[:3]:
        r, x = a[i], b[i]
        ref = clean(r["reference"]) or "Not in the handbook, so the right reply is “I don’t know”."
        cards.append(
            f'<article class="card"><span class="tag">{r["type"].replace("_", "-")}</span><h3>{html.escape(r["question"])}</h3>'
            f'<div class="ans bad"><b>Before</b><p class="clamp">{clean(r["answer"])}</p></div>'
            f'<div class="ans good"><b>After</b><p class="clamp">{clean(x["answer"])}</p></div>'
            f'<details><summary>Full answers and reference</summary><p class="full"><b>Before:</b> {clean(r["answer"])}</p>'
            f'<p class="full"><b>After:</b> {clean(x["answer"])}</p><p class="full"><b>Reference:</b> {ref}</p></details></article>')
    return "".join(cards)


CSS = """
:root{--bg:#fbfaf6;--fg:#1d1d1b;--muted:#6a6a64;--line:#e5e3dc;--card:#fff;--good:#2f8a5f;--bad:#d1553a;--mid:#c98a14;--idk:#c9c6bd;--accent:#2f6a9a}
@media (prefers-color-scheme:dark){:root{--bg:#121311;--fg:#ecebe6;--muted:#a3a29b;--line:#2c2d29;--card:#1b1c19;--good:#5fbf8f;--bad:#ec7a5f;--mid:#e3ad4f;--idk:#4a4a45;--accent:#7fb6e0}}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.55 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1080px;margin:0 auto;padding:0 16px 64px}section{padding:56px 0;border-top:1px solid var(--line)}
.hero{padding:40px 0 48px;border:0}.eyebrow{font-size:.75rem;letter-spacing:.12em;text-transform:uppercase;color:var(--accent);font-weight:700;margin:0 0 10px}
h1{font-size:clamp(1.6rem,4vw,2.5rem);line-height:1.15;margin:0 0 28px;max-width:860px;letter-spacing:-.01em}h1 em{font-style:normal;color:var(--bad)}h1 em.g{color:var(--good)}
h2{font-size:clamp(1.3rem,3vw,1.8rem);margin:0 0 6px;letter-spacing:-.01em}.sub{color:var(--muted);margin:0 0 24px;max-width:640px}
.stage{display:grid;grid-template-columns:minmax(0,1.15fr) minmax(0,1fr);gap:32px;align-items:center}
.dots{display:grid;grid-template-columns:repeat(13,1fr);gap:clamp(4px,1vw,8px)}
.dot{aspect-ratio:1;border-radius:50%;background:var(--idk);border:0;padding:0;cursor:pointer;transition:background-color .45s ease,transform .45s ease}
.dot:hover,.dot:focus-visible{transform:scale(1.25);outline:2px solid var(--fg);outline-offset:1px}
.dot.right{background:var(--good)}.dot.partial{background:var(--mid)}.dot.wrong{background:var(--bad)}.dot.idk{background:var(--idk)}.dot.held{background:var(--accent)}
.dot.flip{animation:pop .5s ease}@keyframes pop{50%{transform:scale(1.35)}}
.legend{display:flex;flex-wrap:wrap;gap:6px 16px;font-size:.8rem;color:var(--muted);margin-top:14px}.legend i{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px;vertical-align:-1px}
.ver{font-size:.8rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:var(--accent)}.change{font-size:1.15rem;line-height:1.35;margin:6px 0 18px;min-height:3.2em}
.tally{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:20px}.tally div{border-left:3px solid;padding-left:10px}
.tally b{display:block;font-size:clamp(1.6rem,4vw,2.3rem);line-height:1.1;font-variant-numeric:tabular-nums}.tally span{font-size:.8rem;color:var(--muted)}
.t-right{border-color:var(--good)!important}.t-wrong{border-color:var(--bad)!important}.t-idk{border-color:var(--idk)!important}
.steps{display:flex;gap:6px;flex-wrap:wrap;align-items:center}.step{font:inherit;font-size:.8rem;font-weight:600;width:38px;height:34px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer}
.step[aria-pressed=true]{background:var(--fg);color:var(--bg);border-color:var(--fg)}.play{font:inherit;font-size:.8rem;height:34px;padding:0 12px;border-radius:8px;border:1px solid var(--line);background:transparent;color:var(--muted);cursor:pointer}
.kpis{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:40px}.kpi{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px}
.kpi b{display:block;font-size:1.35rem;font-variant-numeric:tabular-nums}.kpi span{font-size:.8rem;color:var(--muted)}
.pills{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:18px}.pill{font:inherit;font-size:.85rem;padding:6px 14px;border-radius:99px;border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer}
.pill[aria-pressed=true]{border-color:var(--accent);color:var(--accent);font-weight:600}
.slider{margin:4px 0 22px}.slider label{display:block;font-size:1.05rem;margin-bottom:8px}.slider b{font-size:1.5rem;font-variant-numeric:tabular-nums;color:var(--accent)}
.slider input{width:100%;accent-color:var(--accent)}.ends{display:flex;justify-content:space-between;font-size:.75rem;color:var(--muted)}
.big3{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}.big3 b{display:block;font-size:clamp(1.5rem,4vw,2.1rem);font-variant-numeric:tabular-nums}.big3 span{font-size:.8rem;color:var(--muted)}
.fine{font-size:.75rem;color:var(--muted);margin-top:14px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px}.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px}
.card h3{font-size:1rem;line-height:1.35;margin:8px 0 14px}.tag{font-size:.7rem;font-weight:700;color:var(--accent);text-transform:uppercase;letter-spacing:.06em}
.ans{border-left:3px solid;padding:2px 0 2px 10px;margin-bottom:10px}.ans b{font-size:.7rem;text-transform:uppercase;letter-spacing:.06em}.ans.bad{border-color:var(--bad)}.ans.good{border-color:var(--good)}
.ans.bad b{color:var(--bad)}.ans.good b{color:var(--good)}.ans p{margin:2px 0 0;font-size:.9rem}
.clamp{display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;overflow:hidden}
details>summary{cursor:pointer;color:var(--accent);font-size:.85rem;font-weight:600}.full{font-size:.85rem;white-space:pre-wrap}
.layer{background:var(--card);border:1px solid var(--line);border-radius:12px;margin-bottom:12px}.layer>summary{padding:16px 18px;font-size:1rem;color:var(--fg);list-style:none;display:flex;justify-content:space-between}
.layer>summary::after{content:"+";color:var(--muted);font-size:1.2rem}.layer[open]>summary::after{content:"\\2212"}.layer>summary::-webkit-details-marker{display:none}
.layer>div{padding:0 18px 18px}.layer summary span{color:var(--muted);font-weight:400;font-size:.85rem;margin-left:8px}
.table-wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:.82rem}th,td{padding:7px 9px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{font-weight:600;color:var(--muted)}td{font-variant-numeric:tabular-nums}.table-wrap td:nth-child(2){min-width:220px}.ci{display:block;color:var(--muted);font-size:.72rem}
.gates{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}.gate h3{font-size:.95rem;margin:0}.note{color:var(--muted);font-size:.82rem}
.controls{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px;margin-bottom:12px}
.controls label{font-size:.75rem;color:var(--muted);display:flex;flex-direction:column;gap:4px}
.controls select,.controls input{font:inherit;font-size:.85rem;padding:7px 8px;border-radius:6px;border:1px solid var(--line);background:var(--bg);color:var(--fg);min-width:0}
.count{font-size:.85rem;color:var(--muted);margin:0 0 8px}.qlist{display:grid;gap:8px}
.qitem{border:1px solid var(--line);border-radius:8px;background:var(--bg)}.qitem summary{cursor:pointer;padding:10px 12px;display:grid;grid-template-columns:1fr auto;gap:8px;align-items:start;list-style:none;color:var(--fg);font-weight:400}
.qitem summary::-webkit-details-marker{display:none}.qitem[open] summary{border-bottom:1px solid var(--line)}.qitem.hit{outline:2px solid var(--accent)}
.qtext{font-size:.9rem}.qid{color:var(--muted);font-size:.75rem;margin-right:6px;font-variant-numeric:tabular-nums}.qtext .tag{margin-right:6px}
.badges{display:flex;gap:4px;white-space:nowrap}.b{font-size:.7rem;font-weight:600;padding:2px 7px;border-radius:99px;border:1px solid}
.b.good{color:var(--good);border-color:var(--good)}.b.bad{color:var(--bad);border-color:var(--bad)}.b.mid{color:var(--mid);border-color:var(--mid)}
.qbody{padding:12px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:16px}.pair p{margin:4px 0 0;font-size:.88rem;white-space:pre-wrap}.pair h4{margin:0;font-size:.75rem;color:var(--muted)}
.why{font-size:.78rem;color:var(--muted);margin-top:6px}.ref{font-size:.82rem;color:var(--muted);margin:10px 0 0}.flag{font-size:.7rem;color:var(--mid);margin-left:6px}
.more{margin-top:10px;font:inherit;font-size:.85rem;padding:7px 14px;border-radius:6px;border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer}
ul{padding-left:20px}a{color:var(--accent)}footer{padding-top:32px;border-top:1px solid var(--line);color:var(--muted);font-size:.8rem;display:flex;gap:16px;flex-wrap:wrap}
@media (max-width:760px){.stage{grid-template-columns:1fr;gap:20px}.kpis,.big3{grid-template-columns:1fr 1fr 1fr}.kpi{padding:12px}.kpi b{font-size:1.05rem}.pair{grid-template-columns:1fr}.qitem summary{grid-template-columns:1fr}.badges{flex-wrap:wrap}.change{min-height:0}}
@media (prefers-reduced-motion:reduce){.dot{transition:none}.dot.flip{animation:none}html{scroll-behavior:auto}}
"""


def main() -> None:
    runs = [(n, json.loads((RESULTS / n / "metrics.json").read_text(encoding="utf-8")), _rows(n))
            for n in ORDER if (RESULTS / n / "metrics.json").exists()]
    first, last = runs[0], runs[-1]
    m0, m1 = first[1], last[1]
    w0, w1 = wrong_shown(first[2]), wrong_shown(last[2])
    kpis = [
        (f'{_pct(m0["answer_accuracy"]["value"])} → {_pct(m1["answer_accuracy"]["value"])}', "answers correct"),
        (f'{_pct(m0["unanswerable_refusal"]["value"])} → {_pct(m1["unanswerable_refusal"]["value"])}', "say “I don’t know” when they should"),
        (f'{_pct(stat(m0, "multi_hop")["value"])} → {_pct(stat(m1, "multi_hop")["value"])}', "comparison questions right"),
    ]
    labels = {"v3_rerank": "v3 · plain prompt", "v4_grounded_prompt": "v4 · grounded prompt", "v7_decompose": "v7 · final"}
    gates = [dict(g, label=labels.get(src, src)) for src in ("v4_grounded_prompt", "v7_decompose", "v3_rerank")
             if (g := gate_grid(src))]
    data = explorer_data(runs)
    data["steps"] = {n: STEPS.get(n, m["config"]["description"]) for n, m, _ in runs}
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RAG Eval Kit · Why the chatbot was wrong</title><meta name="description" content="A RAG assistant audited and fixed one measured change at a time.">
<style>{CSS}</style></head><body><main>

<section class="hero">
<p class="eyebrow">RAG audit · case study</p>
<h1>A handbook chatbot gave a confident wrong answer to <em>1 in {round(1 / w0)}</em> questions. Seven measured changes later: <em class="g">1 in {round(1 / w1)}</em>.</h1>
<div class="stage">
<div><div class="dots" id="dots" role="img" aria-label="104 test questions, coloured by how the assistant answered"></div>
<div class="legend"><span><i style="background:var(--good)"></i>right</span><span><i style="background:var(--mid)"></i>partly right</span>
<span><i style="background:var(--bad)"></i>wrong or made up</span><span><i style="background:var(--idk)"></i>said “I don’t know” but could have answered</span></div></div>
<div>
<div class="ver" id="ver"></div><p class="change" id="change"></p>
<div class="tally"><div class="t-right"><b id="t-right">0</b><span>right</span></div><div class="t-wrong"><b id="t-wrong">0</b><span>wrong or made up</span></div><div class="t-idk"><b id="t-idk">0</b><span>missed</span></div></div>
<div class="steps" id="steps" role="group" aria-label="Version"></div>
</div></div>
<div class="kpis">{"".join(f'<div class="kpi"><b>{v}</b><span>{l}</span></div>' for v, l in kpis)}</div>
</section>

<section>
<h2>Fewer mistakes, or more answers? You choose.</h2>
<p class="sub">A checker reads each answer against its sources. Anything it doubts goes to a person instead of the user.</p>
<div class="pills" id="gate-src" role="group" aria-label="Assistant version"></div>
<div class="stage">
<div><div class="dots" id="gdots" role="img" aria-label="The same 104 questions with the safety gate applied"></div>
<div class="legend"><span><i style="background:var(--good)"></i>right, shown</span><span><i style="background:var(--bad)"></i>wrong, shown</span>
<span><i style="background:var(--accent)"></i>held back for a person</span><span><i style="background:var(--idk)"></i>“I don’t know”</span></div></div>
<div>
<div class="slider"><label for="alpha">Allow at most <b id="alpha-out">3%</b> wrong answers</label>
<input type="range" id="alpha" min="0.01" max="0.15" step="0.005" value="0.03"><div class="ends"><span>stricter</span><span>looser</span></div></div>
<div class="big3"><div><b id="g-wrong"></b><span>wrong answers shown</span></div><div><b id="g-answered"></b><span>answered directly</span></div><div><b id="g-held"></b><span>sent to a person</span></div></div>
<p class="fine" id="g-base"></p>
</div></div>
</section>

<section>
<h2>What changed in the answers</h2>
<p class="sub">Three of the questions the baseline got wrong and the final version got right.</p>
<div class="cards">{_examples(first[2], last[2])}</div>
</section>

<section>
<h2>Go deeper</h2>
<p class="sub">Everything behind the numbers, for whoever wants to check.</p>

<details class="layer" id="explorer"><summary>Check every answer yourself<span>104 questions, any two versions side by side</span></summary><div>
<div class="controls">
<label>Left<select id="left"></select></label><label>Right<select id="right"></select></label>
<label>Type<select id="type"><option value="">all types</option></select></label>
<label>Show<select id="show"><option value="">all</option><option value="fixed">fixed (left wrong → right right)</option>
<option value="broke">broke (left right → right wrong)</option><option value="wrong">still not right on the right</option>
<option value="flag">flagged in audit (⚑)</option></select></label>
<label>Search<input id="search" type="search" placeholder="leave, pension, q076…"></label>
</div>
<p class="count" id="count"></p><div class="qlist" id="qlist"></div><button type="button" class="more" id="more">Show more</button>
</div></details>

<details class="layer"><summary>All metrics with confidence intervals<span>7 versions × 9 measures</span></summary><div>
<p class="note">95% bootstrap intervals under each value. Differences inside overlapping intervals are not established improvements.</p>
{_table(runs)}</div></details>

<details class="layer"><summary>The safety gate in detail<span>method and published targets</span></summary><div>
<p class="note">The checker is a third model family (neither the assistant nor the grader). Its threshold is set with conformal risk control on
half the questions and tested on the other half, averaged over 1000 random splits. The guarantee holds on average, not on every split;
with 52 calibration questions single splits vary, as the 90th-percentile column shows. The dots above show as many held-back answers as the average split, choosing the questions held back most often across splits.</p>
<div class="gates">{_gate()}</div></div></details>

<details class="layer"><summary>How it was measured, and the limits<span>read before trusting the numbers</span></summary><div><ul>
<li>104 questions over 94 pages of the public GitLab Handbook: facts, paraphrases, numbers, cross-country comparisons, and 23 questions the handbook cannot answer. 12 unsound drafts were removed, with reasons published.</li>
<li>Assistant <code>{m1["generator"]}</code>; grader <code>{m1["judge"]}</code>, a different model family. The grader was checked against blind grades from an independent LLM reviewer of a third family (agreement κ = 0.93, n = 42). That is a cross-check between models, not a human gold standard.</li>
<li>Only 14 comparison questions, so that number has a wide interval.</li>
<li>One corpus, one assistant model. Your system will behave differently, which is the reason to measure it.</li>
<li>Every model response is committed, so all numbers replay offline without an API key.</li>
</ul></div></details>
</section>

<footer><span>RAG Eval Kit</span><a href="https://github.com/babuilabs/rag-eval-kit">Code</a>
<a href="https://github.com/babuilabs/rag-eval-kit/blob/main/docs/CASE_STUDY.md">Case study</a>
<span>Corpus: GitLab Handbook, CC BY-SA 4.0, © GitLab B.V. · Code: MIT</span></footer>
</main>
{_json_script("gate-data", gates)}
{_json_script("explorer-data", data)}
<script>{JS.read_text(encoding="utf-8")}</script>
</body></html>
"""
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(page, encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
