"""Build the static HTML report (docs/index.html) from the finished runs.

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


def _bars(runs: list[tuple[str, dict, list[dict]]]) -> str:
    """Two horizontal bar groups: answer accuracy (higher is better) and wrong answers shown."""
    def group(title: str, note: str, values: list[tuple[str, float]], scale: float, cls: str) -> str:
        items = "".join(
            f'<div class="bar-row"><span class="bar-label">{html.escape(SHORT.get(n, n))}</span>'
            f'<span class="bar-track"><span class="bar {cls}" style="width:{v / scale * 100:.1f}%"></span></span>'
            f'<span class="bar-value">{_pct(v, 1 if scale < 0.5 else 0)}</span></div>'
            for n, v in values)
        return f'<div class="chart"><h3>{title}</h3><p class="note">{note}</p>{items}</div>'

    acc = [(n, m["answer_accuracy"]["value"]) for n, m, _ in runs]
    wrong = [(n, wrong_shown(rows)) for n, _, rows in runs]
    return (group("Answer accuracy", "Questions the handbook answers (n=81). Partial answers count half.", acc, 1.0, "good")
            + group("Wrong answers shown", "Share of all 104 questions where a wrong answer reached the user. Lower is better.",
                    wrong, max(v for _, v in wrong) * 1.15, "bad"))


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
        for idx in perms:
            cal, test = idx[:half], idx[half:]
            taus = np.unique(np.concatenate([[0.0, 1.0 + 1e-9], p[cal][p[cal] >= 0]]))
            cal_risk = (lo[cal][:, None] * (p[cal][:, None] >= taus[None, :])).sum(0) / half
            ok = (half * cal_risk + 1) / (half + 1) <= alpha
            tau = taus[ok.argmax()] if ok.any() else 1.0 + 1e-9
            shown = p[test] >= tau
            wrong.append((lo[test] * shown).mean())
            answered.append(shown.mean())
            right.append((shown & (lo[test] == 0)).mean())
        wrong = np.sort(wrong)
        out[str(alpha)] = {"wrong": round(float(np.mean(wrong)), 4), "wrong_p90": round(float(wrong[int(0.9 * splits)]), 4),
                           "answered": round(float(np.mean(answered)), 4), "right": round(float(np.mean(right)), 4)}
    shown_all = p >= 0
    return {"source": source, "n": n, "calibration_n": half, "splits": splits,
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


def _examples(before: list[dict], after: list[dict]) -> str:
    """One question per type where the baseline failed and the final version got it right."""
    a = {r["id"]: r for r in after}
    picked, seen = [], set()
    for r in before:
        x = a.get(r["id"])
        if x and r["type"] not in seen and r["id"] not in WEAK and r["score"] == 0 and x["score"] == 1:
            seen.add(r["type"])
            picked.append((r, x))
    def cut(s: str) -> str:
        s = s.replace("**", "")
        return html.escape(s if len(s) < 420 else s[:420].rsplit(" ", 1)[0] + " …")
    return "".join(
        f'<div class="example"><div class="q"><span class="tag">{r["type"].replace("_", "-")}</span>{html.escape(r["question"])}</div>'
        f'<div class="pair"><div><h4>Baseline</h4><p>{cut(r["answer"])}</p></div>'
        f'<div><h4>Final</h4><p>{cut(x["answer"])}</p></div></div>'
        f'<p class="ref"><b>Reference:</b> {cut(r["reference"]) or "Not in the handbook. The right reply is “I don’t know”."}</p></div>'
        for r, x in picked)


CSS = """
:root{--bg:#fbfbf9;--fg:#1d1d1b;--muted:#6a6a64;--line:#e3e2dc;--card:#fff;--good:#2f7d5b;--bad:#c2553a;--accent:#2f5d7d;--mid:#a86a12}
@media (prefers-color-scheme:dark){:root{--bg:#151614;--fg:#ecebe6;--muted:#a3a29b;--line:#2f302c;--card:#1d1e1b;--good:#5fb48c;--bad:#e07a5f;--accent:#7fb0d4;--mid:#e0a84a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.6 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1040px;margin:0 auto;padding:40px 16px 80px}h1{font-size:2rem;line-height:1.2;margin:0 0 8px}
h2{margin:48px 0 12px;font-size:1.3rem}h3{margin:0 0 4px;font-size:1rem}h4{margin:0 0 4px;font-size:.8rem;text-transform:uppercase;letter-spacing:.05em;color:var(--muted)}
.lede{color:var(--muted);max-width:720px}.note{color:var(--muted);font-size:.85rem;margin:0 0 12px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin:28px 0}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px}.kpi b{display:block;font-size:1.6rem}
.kpi span{color:var(--muted);font-size:.85rem}.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:24px}
.chart,.gate,.example{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px}
.bar-row{display:grid;grid-template-columns:160px 1fr 56px;align-items:center;gap:8px;font-size:.85rem;margin:6px 0}
.bar-track{background:var(--line);border-radius:4px;height:12px;overflow:hidden}.bar{display:block;height:100%;border-radius:4px}
.bar.good{background:var(--good)}.bar.bad{background:var(--bad)}.bar-value{text-align:right;font-variant-numeric:tabular-nums}
.table-wrap{overflow-x:auto;border:1px solid var(--line);border-radius:10px;background:var(--card)}
table{border-collapse:collapse;width:100%;font-size:.85rem}th,td{padding:8px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{font-weight:600;color:var(--muted)}td{font-variant-numeric:tabular-nums}.table-wrap td:nth-child(2){min-width:240px}.ci{display:block;color:var(--muted);font-size:.75rem}
.gates{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}.examples{display:grid;gap:16px}
.q{font-weight:600;margin-bottom:10px}.tag{font-size:.7rem;font-weight:600;color:var(--accent);border:1px solid var(--accent);border-radius:99px;padding:1px 8px;margin-right:8px;vertical-align:middle}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:16px}.pair p{margin:0;font-size:.9rem}.ref{font-size:.85rem;color:var(--muted);margin:10px 0 0}
@media (max-width:640px){.pair{grid-template-columns:1fr}.bar-row{grid-template-columns:110px 1fr 48px}}
.pills{display:flex;flex-wrap:wrap;gap:8px;margin:0 0 14px}.pill{font:inherit;font-size:.85rem;padding:6px 12px;border-radius:99px;border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer}
.pill[aria-pressed=true]{border-color:var(--accent);color:var(--accent);font-weight:600}
.panel{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:18px}
.slider-row{display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin:6px 0 16px}.slider-row input{flex:1;min-width:200px;accent-color:var(--accent)}
.target{font-size:1.05rem}.target b{font-size:1.4rem;font-variant-numeric:tabular-nums}
.stack{display:flex;height:28px;border-radius:6px;overflow:hidden;margin:6px 0 10px;background:var(--line)}.stack span{height:100%;transition:width .25s}
.s-right{background:var(--good)}.s-wrong{background:var(--bad)}.s-held{background:var(--accent)}.s-idk{background:var(--muted);opacity:.45}
.legend{display:flex;flex-wrap:wrap;gap:6px 18px;font-size:.85rem}.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:middle}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-top:16px}.stat{border-top:1px solid var(--line);padding-top:8px}
.stat b{display:block;font-size:1.3rem;font-variant-numeric:tabular-nums}.stat span{color:var(--muted);font-size:.8rem}
.controls{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin-bottom:12px}
.controls label{font-size:.75rem;color:var(--muted);display:flex;flex-direction:column;gap:4px}
.controls select,.controls input{font:inherit;font-size:.85rem;padding:7px 8px;border-radius:6px;border:1px solid var(--line);background:var(--bg);color:var(--fg);min-width:0}
.count{font-size:.85rem;color:var(--muted);margin:0 0 8px}.qlist{display:grid;gap:8px}
.qitem{background:var(--card);border:1px solid var(--line);border-radius:8px}.qitem summary{cursor:pointer;padding:10px 12px;display:grid;grid-template-columns:1fr auto;gap:8px;align-items:start;list-style:none}
.qitem summary::-webkit-details-marker{display:none}.qitem[open] summary{border-bottom:1px solid var(--line)}
.qtext{font-size:.9rem}.qid{color:var(--muted);font-size:.75rem;margin-right:6px;font-variant-numeric:tabular-nums}
.badges{display:flex;gap:4px;white-space:nowrap}.b{font-size:.7rem;font-weight:600;padding:2px 7px;border-radius:99px;border:1px solid;text-transform:none;letter-spacing:0}
.b.good{color:var(--good);border-color:var(--good)}.b.bad{color:var(--bad);border-color:var(--bad)}.b.mid{color:var(--mid);border-color:var(--mid)}
.qbody{padding:12px}.qbody .pair p{white-space:pre-wrap}.why{font-size:.8rem;color:var(--muted);margin-top:6px}.flag{font-size:.7rem;color:var(--mid);margin-left:6px}
.more{margin-top:10px;font:inherit;font-size:.85rem;padding:7px 14px;border-radius:6px;border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer}
details.fixed summary{cursor:pointer;color:var(--muted);font-size:.85rem;margin-top:14px}
@media (max-width:640px){.qitem summary{grid-template-columns:1fr}.badges{flex-wrap:wrap}}
ul{padding-left:20px}a{color:var(--accent)}footer{margin-top:56px;color:var(--muted);font-size:.8rem}
"""


def main() -> None:
    runs = [(n, json.loads((RESULTS / n / "metrics.json").read_text(encoding="utf-8")), _rows(n))
            for n in ORDER if (RESULTS / n / "metrics.json").exists()]
    first, last = runs[0], runs[-1]
    m0, m1 = first[1], last[1]
    kpis = [
        (f'{_pct(m0["answer_accuracy"]["value"])} → {_pct(m1["answer_accuracy"]["value"])}', "answer accuracy"),
        (f'{_pct(wrong_shown(first[2]), 1)} → {_pct(wrong_shown(last[2]), 1)}', "of questions got a wrong answer"),
        (f'{_pct(m0["unanswerable_refusal"]["value"])} → {_pct(m1["unanswerable_refusal"]["value"])}', "say “don’t know” when the answer isn’t there"),
        (f'{_pct(stat(m0, "multi_hop")["value"])} → {_pct(stat(m1, "multi_hop")["value"])}', "on comparison questions (n=14)"),
    ]
    labels = {"v3_rerank": "v3 reranker, plain prompt", "v4_grounded_prompt": "v4 grounded prompt",
              "v7_decompose": "v7 split comparisons"}
    gates = [dict(g, label=labels.get(src, src)) for src in ("v4_grounded_prompt", "v7_decompose", "v3_rerank")
             if (g := gate_grid(src))]
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RAG Eval Kit report</title><meta name="description" content="Measuring and fixing a RAG assistant one change at a time, with confidence intervals.">
<style>{CSS}</style></head><body><main>
<h1>Why does the handbook chatbot get it wrong?</h1>
<p class="lede">A typical tutorial-style RAG assistant over a company handbook, measured on 104 questions, then fixed one change at a time.
Every number below comes from the runs in this repository and can be replayed offline from the committed model responses.</p>
<div class="kpis">{"".join(f'<div class="kpi"><b>{v}</b><span>{l}</span></div>' for v, l in kpis)}</div>

<h2>What each change did</h2>
<div class="charts">{_bars(runs)}</div>

<h2>All metrics</h2>
<p class="note">95% bootstrap confidence intervals under each value. v5 is the safe gate below; it filters answers instead of changing them.</p>
{_table(runs)}

<h2>Safe gate: choose how many wrong answers you can live with</h2>
<p class="note">A third model checks whether every claim in an answer is supported by the passages. Pick a target below: the threshold is
set with conformal risk control on half the questions and tested on the other half, averaged over 1000 random splits. The guarantee holds
on average, not for every split; with 52 calibration questions single splits vary, which the 90th-percentile figure shows.</p>
<div class="panel">
<div class="pills" id="gate-src" role="group" aria-label="Assistant version"></div>
<div class="slider-row"><span class="target">Target: at most <b id="alpha-out">3%</b> wrong answers shown</span>
<input type="range" id="alpha" min="0.01" max="0.15" step="0.005" value="0.03" aria-label="Target rate of wrong answers"></div>
<div class="stack" aria-hidden="true"><span class="s-right" id="s-right"></span><span class="s-wrong" id="s-wrong"></span><span class="s-held" id="s-held"></span><span class="s-idk" id="s-idk"></span></div>
<div class="legend"><span><i class="s-right"></i>fully right, shown</span><span><i class="s-wrong"></i>wrong or partial, shown</span>
<span><i class="s-held"></i>held back, sent to a person</span><span><i class="s-idk"></i>assistant said “I don’t know”</span></div>
<div class="stats"><div class="stat"><b id="g-wrong"></b><span>wrong answers shown (mean; partial counts half)</span></div>
<div class="stat"><b id="g-p90"></b><span>wrong shown, 90th-percentile split</span></div>
<div class="stat"><b id="g-answered"></b><span>questions answered directly</span></div>
<div class="stat"><b id="g-held"></b><span>sent to a person</span></div></div>
<p class="note" id="g-base" style="margin-top:12px"></p>
<details class="fixed"><summary>Fixed targets as published in the README</summary><div class="gates" style="margin-top:12px">{_gate()}</div></details>
</div>

<h2>Before and after</h2>
<p class="note">One question of each type where the baseline was wrong and the final version was right. Text is from the GitLab Handbook (CC BY-SA 4.0).</p>
<div class="examples">{_examples(first[2], last[2])}</div>

<h2>Explore all 104 questions</h2>
<p class="note">Compare any two versions question by question, with the judge's reasoning. "Fixed" and "broke" are relative to the left version.
Questions flagged in the audit as ambiguous are marked ⚑.</p>
<div class="controls">
<label>Left<select id="left"></select></label><label>Right<select id="right"></select></label>
<label>Type<select id="type"><option value="">all types</option></select></label>
<label>Show<select id="show"><option value="">all</option><option value="fixed">fixed (left wrong → right right)</option>
<option value="broke">broke (left right → right wrong)</option><option value="wrong">still not right on the right</option>
<option value="flag">flagged in audit</option></select></label>
<label>Search<input id="search" type="search" placeholder="leave, pension, q076…"></label>
</div>
<p class="count" id="count"></p>
<div class="qlist" id="qlist"></div>
<button type="button" class="more" id="more">Show more</button>

<h2>Read this before trusting the numbers</h2>
<ul>
<li>One corpus (94 handbook pages), one generator (<code>{m1["generator"]}</code>), one judge (<code>{m1["judge"]}</code>). Results on your data will differ; that is the point of running the kit on it.</li>
<li>Correctness is graded by an LLM judge. Its verdicts were checked against blind grades from an independent LLM reviewer of a different model family from the generator and the judge (agreement κ = 0.93, n = 42). That is a check, not a human gold standard.</li>
<li>The multi-hop subset has only 14 questions; its intervals are wide.</li>
<li>Version-to-version differences inside overlapping intervals are not established improvements.</li>
</ul>
<footer>Corpus: GitLab Handbook, CC BY-SA 4.0, © GitLab B.V. · Code: MIT</footer>
</main>
{_json_script("gate-data", gates)}
{_json_script("explorer-data", explorer_data(runs))}
<script>{JS.read_text(encoding="utf-8")}</script>
</body></html>
"""
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(page, encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
