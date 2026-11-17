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
CONTACT = "https://github.com/babuilabs"
REPO = "https://github.com/babuilabs/rag-eval-kit"

# Each step in words a non-specialist can follow. The technical names live in the "Go deeper" layer.
PLAIN = {
    "v0_baseline": "The standard tutorial setup",
    "v1_heading_chunks": "Cut the documents where the topic changes",
    "v2_hybrid": "Add keyword search",
    "v3_rerank": "Re-rank the passages it finds",
    "v4_grounded_prompt": "Make it cite sources and admit when it doesn’t know",
    "v6_more_context": "Let it read six passages instead of three",
    "v7_decompose": "Search each side of a comparison separately",
}
SHORT = {
    "v0_baseline": "Tutorial baseline",
    "v1_heading_chunks": "+ heading-aware chunks",
    "v2_hybrid": "+ hybrid search",
    "v3_rerank": "+ reranker",
    "v4_grounded_prompt": "+ grounded prompt",
    "v6_more_context": "+ 6 passages",
    "v7_decompose": "+ split comparisons",
}
WORDS = {5: "Five", 6: "Six", 7: "Seven", 8: "Eight"}
KIND = {"fact": "Simple fact", "paraphrase": "Reworded question", "numeric": "A number",
        "multi_hop": "Comparison", "unanswerable": "Not in the handbook"}
HERO_ID = "q106"  # unanswerable: the baseline invents a number, the final version says it doesn't know
CARD_IDS = ("q076", "q066")  # one comparison, one fact; a third of another type is filled in below

MARK = """<svg viewBox="0 0 120 140" aria-hidden="true"><defs><clipPath id="nc{k}"><path d="M60 30C86 32 96 60 90 86C86 104 74 114 60 114C46 114 34 104 30 86C24 60 34 32 60 30Z"/></clipPath>
<pattern id="nw{k}" width="9" height="9" patternUnits="userSpaceOnUse" patternTransform="rotate(35)"><line x1="0" y1="0" x2="9" y2="0" stroke="#B9852A" stroke-width="1.6"/><line x1="0" y1="0" x2="0" y2="9" stroke="#C99536" stroke-width="1.1"/></pattern></defs>
<path d="M8 18C40 8 80 8 112 14" fill="none" stroke="#7FC4B0" stroke-width="4.5" stroke-linecap="round"/><path d="M92 12C98 2 108 0 114 2C110 10 102 14 92 12Z" fill="#2E8A73"/>
<path d="M57 13L58 31M63 13L62 31" stroke="#B9852A" stroke-width="2.4" stroke-linecap="round"/>
<path d="M60 30C86 32 96 60 90 86C86 104 74 114 60 114C46 114 34 104 30 86C24 60 34 32 60 30Z" fill="#E3B455"/><rect x="20" y="25" width="80" height="95" fill="url(#nw{k})" clip-path="url(#nc{k})" opacity=".85"/>
<g stroke="#1F5F5B" stroke-width="1.8" stroke-linecap="round"><path d="M48 58L70 52L58 80L48 58M70 52L78 76L58 80L48 96L68 98L78 76"/></g>
<g fill="#1F5F5B"><circle cx="48" cy="58" r="3.6"/><circle cx="70" cy="52" r="3.6"/><circle cx="58" cy="80" r="4.2"/><circle cx="78" cy="76" r="3.6"/><circle cx="48" cy="96" r="3.2"/><circle cx="68" cy="98" r="3.2"/></g>
<path d="M53 110C53 120 54 128 55 134L65 134C66 128 67 120 67 110Z" fill="#D4A245"/><ellipse cx="60" cy="134" rx="5" ry="2.4" fill="#5A3E12"/></svg>"""


def _rows(run: str) -> list[dict]:
    return [json.loads(l) for l in (RESULTS / run / "answers.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]


def wrong_shown(rows: list[dict]) -> float:
    return sum(loss(r) * (not r["verdict"].get("refused")) for r in rows) / len(rows)


def _state(r: dict) -> str:
    """How one answer looks to a user: right, partly right, wrong (or made up), or a refusal it didn't need."""
    refused = bool(r["verdict"].get("refused"))
    if r["type"] == "unanswerable":
        return "right" if refused else "wrong"
    if refused:
        return "idk"
    return "right" if r["score"] >= 1 else "partial" if r["score"] > 0 else "wrong"


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
    return {"versions": [{"id": n, "label": f'Step {i} · {PLAIN.get(n, n)}' if i else "Start · tutorial setup"}
                         for i, (n, _, _) in enumerate(runs)],
            "kinds": KIND, "questions": list(qs.values()), "answers": answers}


def _json_script(element_id: str, data) -> str:
    text = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return f'<script type="application/json" id="{element_id}">{text}</script>'


def _clean(s: str) -> str:
    return html.escape(s.replace("**", ""))


def _first_line(s: str) -> str:
    """The opening sentence or paragraph of an answer: what a user reads before deciding to trust it."""
    para = s.replace("**", "").strip().split("\n")[0].strip()
    return para if len(para) < 220 else para[:200].rsplit(" ", 1)[0] + "…"


def _journey(runs: list[tuple[str, dict, list[dict]]]) -> str:
    """One stacked bar per step: how the 104 answers split, plus confident wrong answers per 100."""
    out = []
    prev = None
    for i, (name, _, rows) in enumerate(runs):
        n = len(rows)
        c = {k: sum(_state(r) == k for r in rows) for k in ("right", "partial", "wrong", "idk")}
        ws = wrong_shown(rows) * 100
        segs = "".join(f'<i class="s-{k}" style="--w:{c[k] / n * 100:.2f}%"></i>' for k in ("right", "partial", "idk", "wrong"))
        worse = prev is not None and ws > prev + 1 and name != "v6_more_context"
        note = '<em class="worse">made it worse</em>' if worse else ""
        out.append(
            f'<li class="jrow{" first" if i == 0 else ""}{" last" if i == len(runs) - 1 else ""}">'
            f'<span class="jn">{i if i else "Start"}</span><span class="jl">{html.escape(PLAIN.get(name, name))}{note}</span>'
            f'<span class="jbar" role="img" aria-label="{c["right"]} right, {c["partial"]} partly right, '
            f'{c["wrong"]} wrong, {c["idk"]} unanswered out of {n}">{segs}</span>'
            f'<span class="jw"><b>{ws:.0f}</b> in 100</span></li>')
        prev = ws
    return "".join(out)


def _bubble(who: str, cls: str, text: str, verdict: str = "") -> str:
    v = f'<span class="verdict">{verdict}</span>' if verdict else ""
    return f'<div class="bubble {cls}"><span class="who">{who}</span><p>{text}</p>{v}</div>'


def _examples(before: list[dict], after: list[dict]) -> str:
    """Three real questions as short chats. Full answers sit behind a toggle."""
    a, b = {r["id"]: r for r in before}, {r["id"]: r for r in after}
    ids = [i for i in CARD_IDS if i in a and i in b and a[i]["score"] < 1 <= b[i]["score"] and i not in WEAK]
    for r in before:  # fall back to the first fixed question of another type
        if len(ids) >= 3:
            break
        x = b.get(r["id"])
        if x and r["id"] not in ids and r["id"] != HERO_ID and r["id"] not in WEAK and r["score"] == 0 \
                and x["score"] == 1 and r["type"] not in {a[i]["type"] for i in ids} | {"unanswerable"}:
            ids.append(r["id"])
    cards = []
    for i in ids[:3]:
        r, x = a[i], b[i]
        ref = _clean(r["reference"]) or "Not in the handbook, so the right reply is “I don’t know”."
        bad = "Made up" if r["type"] == "unanswerable" else "Gave up" if r["verdict"].get("refused") else "Wrong"
        cards.append(
            f'<article class="chat"><span class="kind">{KIND.get(r["type"], r["type"])}</span>'
            + _bubble("Employee", "q", html.escape(r["question"]))
            + _bubble("Before", "bad", html.escape(_first_line(r["answer"])), bad)
            + _bubble("After", "good", html.escape(_first_line(x["answer"])), "Correct")
            + f'<details><summary>Full answers and the handbook’s answer</summary><p class="full"><b>Before:</b> {_clean(r["answer"])}</p>'
              f'<p class="full"><b>After:</b> {_clean(x["answer"])}</p><p class="full"><b>Handbook:</b> {ref}</p></details></article>')
    return "".join(cards)


CSS = """
:root{--ink:#0f2a28;--ink2:#163b38;--teal:#1f5f5b;--leaf:#2e8a73;--straw:#e3b455;--straw2:#b9852a;--cream:#f8f3e8;--paper:#fffdf8;
--fg:#1c2321;--muted:#5f6b67;--line:#e4dccb;--good:#2e8a73;--bad:#d4573b;--mid:#d9a23a;--idk:#cfc8b8;--held:#4f8fc0;
--serif:"Fraunces",Georgia,"Times New Roman",serif;--sans:"Inter",system-ui,-apple-system,"Segoe UI",sans-serif}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--cream);color:var(--fg);font:16px/1.6 var(--sans);-webkit-font-smoothing:antialiased}
.wrap{max-width:1120px;margin:0 auto;padding:0 16px}a{color:inherit}
h1,h2,h3{font-family:var(--serif);font-weight:600;letter-spacing:-.015em;margin:0}
h2{font-size:clamp(1.7rem,4vw,2.6rem);line-height:1.12;max-width:760px}.lede{font-size:1.1rem;color:var(--muted);max-width:640px;margin:14px 0 0}
section{padding:clamp(64px,10vw,112px) 0}.kicker{font:700 .74rem/1 var(--sans);letter-spacing:.16em;text-transform:uppercase;color:var(--straw2);margin:0 0 14px}
.btn{display:inline-flex;align-items:center;gap:8px;font:600 1rem/1 var(--sans);padding:15px 22px;border-radius:999px;text-decoration:none;transition:transform .2s ease,background .2s ease}
.btn:hover{transform:translateY(-2px)}.btn.primary{background:var(--straw);color:var(--ink)}.btn.primary:hover{background:#ecc46f}
.btn.ghost{color:#e9f1ee;border:1px solid rgba(233,241,238,.3)}.btn.ghost:hover{border-color:#e9f1ee}
/* hero */
.hero{position:relative;overflow:hidden;background:radial-gradient(1200px 600px at 85% 20%,#24524d 0%,transparent 60%),var(--ink);color:#e9f1ee;padding:0 0 clamp(56px,8vw,96px);min-height:100svh;display:flex;flex-direction:column}
.hero::before{content:"";position:absolute;inset:0;opacity:.06;background:repeating-linear-gradient(35deg,var(--straw) 0 1px,transparent 1px 11px),repeating-linear-gradient(125deg,var(--straw) 0 1px,transparent 1px 11px);pointer-events:none}
nav{position:relative;display:flex;align-items:center;justify-content:space-between;gap:16px;padding:22px 0}
.brand{display:flex;align-items:center;gap:10px;font:600 1.15rem/1 var(--serif);text-decoration:none}.brand svg{width:30px;height:35px}
.navlinks{display:flex;gap:22px;align-items:center;font-size:.92rem}.navlinks a{text-decoration:none;opacity:.8}.navlinks a:hover{opacity:1}
.hero-grid{position:relative;flex:1;display:grid;grid-template-columns:minmax(0,1.1fr) minmax(0,1fr);gap:clamp(32px,5vw,72px);align-items:center;padding-top:clamp(24px,5vw,48px)}
.hero h1{font-size:clamp(2.3rem,6vw,4.3rem);line-height:1.04;font-weight:600}.hero h1 em{font-style:italic;color:var(--straw)}
.hero .lede{color:#b9cbc6;font-size:clamp(1.05rem,2vw,1.2rem);margin-top:22px;max-width:540px}
.proof{display:flex;align-items:baseline;gap:14px;flex-wrap:wrap;margin:30px 0 34px}.proof b{font:600 clamp(2rem,5vw,3rem)/1 var(--serif)}
.proof .from{color:#f08a6f}.proof .to{color:#7fd6b4}.proof .arrow{color:var(--straw);font-size:1.6rem}.proof span.cap{flex-basis:100%;font-size:.88rem;color:#9fb5af;margin-top:4px}
.ctas{display:flex;gap:12px;flex-wrap:wrap}
.trust{position:relative;display:flex;gap:10px 28px;flex-wrap:wrap;font-size:.85rem;color:#9fb5af;margin-top:clamp(40px,6vw,64px);padding-top:22px;border-top:1px solid rgba(233,241,238,.12)}
.trust span::before{content:"✓";color:var(--straw);margin-right:8px}
.stage{position:relative}.stage .bigmark{position:absolute;width:min(420px,90%);right:-40px;top:50%;transform:translateY(-50%);opacity:.08;pointer-events:none}
.phone{position:relative;background:var(--paper);color:var(--fg);border-radius:22px;padding:18px;box-shadow:0 30px 80px -20px rgba(0,0,0,.55),0 0 0 1px rgba(255,255,255,.06);max-width:470px;margin-left:auto}
.phone header{display:flex;align-items:center;gap:10px;padding:2px 4px 14px;border-bottom:1px solid var(--line);margin-bottom:14px;font-size:.85rem;font-weight:600}
.phone header i{width:9px;height:9px;border-radius:50%;background:var(--good);box-shadow:0 0 0 3px rgba(46,138,115,.18)}
.phone header small{margin-left:auto;font-weight:400;color:var(--muted)}
.bubble{position:relative;border-radius:16px;padding:11px 14px;margin:0 0 12px;font-size:.93rem;line-height:1.45;max-width:92%}
.bubble p{margin:0}.bubble .who{display:block;font-size:.68rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;margin-bottom:3px;opacity:.7}
.bubble.q{background:var(--teal);color:#fff;margin-left:auto;border-bottom-right-radius:4px}
.bubble.bad{background:#fdece6;border:1px solid #f3c3b5;border-bottom-left-radius:4px}.bubble.bad .who{color:var(--bad);opacity:1}
.bubble.good{background:#e7f4ee;border:1px solid #b9dccb;border-bottom-left-radius:4px}.bubble.good .who{color:var(--good);opacity:1}
.verdict{display:inline-block;margin-top:8px;font-size:.72rem;font-weight:700;padding:3px 9px;border-radius:99px;color:#fff}
.bad .verdict{background:var(--bad)}.good .verdict{background:var(--good)}
.stamp{position:absolute;right:-6px;top:-12px;transform:rotate(6deg);font:700 .72rem/1 var(--sans);letter-spacing:.08em;text-transform:uppercase;color:var(--bad);border:2px solid var(--bad);background:var(--paper);padding:5px 8px;border-radius:6px}
.phone .divider{display:flex;align-items:center;gap:10px;font-size:.72rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:var(--straw2);margin:16px 0 12px}
.phone .divider::before,.phone .divider::after{content:"";flex:1;height:1px;background:var(--line)}
.js .anim .phone>*{opacity:0;transform:translateY(10px)}.js .anim .stamp{opacity:0}.play .phone>*{animation:rise .6s ease forwards}
.play .phone>:nth-child(2){animation-delay:.4s}.play .phone>:nth-child(3){animation-delay:1.4s}.play .phone>:nth-child(4){animation-delay:2.7s}.play .phone>:nth-child(5){animation-delay:3.2s}
.play .stamp{animation:thud .35s ease 2.2s forwards}
@keyframes rise{to{opacity:1;transform:none}}@keyframes thud{0%{opacity:0;transform:rotate(6deg) scale(1.8)}100%{opacity:1;transform:rotate(6deg) scale(1)}}
/* before/after grid */
.ba{background:var(--paper)}.ba-grid{display:grid;grid-template-columns:minmax(0,1.2fr) minmax(0,1fr);gap:clamp(28px,5vw,64px);align-items:center;margin-top:44px}
.toggle{display:inline-flex;background:var(--cream);border:1px solid var(--line);border-radius:999px;padding:4px;margin-bottom:22px}
.toggle button{font:600 .92rem/1 var(--sans);border:0;background:transparent;color:var(--muted);padding:11px 18px;border-radius:999px;cursor:pointer}
.toggle button[aria-pressed=true]{background:var(--ink);color:#fff}
.dots{display:grid;grid-template-columns:repeat(13,1fr);gap:clamp(5px,1vw,9px)}
.dot{aspect-ratio:1;border-radius:50%;background:var(--idk);border:0;padding:0;cursor:pointer;transition:background-color .5s ease,transform .5s ease}
.dot:hover,.dot:focus-visible{transform:scale(1.3);outline:2px solid var(--ink);outline-offset:1px}
.dot.right{background:var(--good)}.dot.partial{background:var(--mid)}.dot.wrong{background:var(--bad)}.dot.idk{background:var(--idk)}
.dot.flip{animation:pop .5s ease}@keyframes pop{50%{transform:scale(1.4)}}
.legend{display:flex;flex-wrap:wrap;gap:6px 18px;font-size:.82rem;color:var(--muted);margin-top:16px}.legend i{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:7px;vertical-align:-1px}
.tally{display:grid;gap:22px}.tally div{display:grid;grid-template-columns:auto 1fr;gap:4px 18px;align-items:baseline}
.tally b{font:600 clamp(2.6rem,6vw,3.6rem)/1 var(--serif);font-variant-numeric:tabular-nums;min-width:2.2ch;text-align:right}
.tally span{font-size:1.02rem}.tally small{grid-column:2;color:var(--muted);font-size:.85rem}
.tally .t-right b{color:var(--good)}.tally .t-wrong b{color:var(--bad)}.tally .t-idk b{color:#9d957f}
.hint{font-size:.82rem;color:var(--muted);margin-top:26px}
/* journey */
.journey{list-style:none;margin:44px 0 0;padding:0;display:grid;gap:4px}
.jrow{display:grid;grid-template-columns:44px minmax(0,1.15fr) minmax(0,1fr) 92px;gap:16px;align-items:center;padding:13px 0;border-bottom:1px solid var(--line)}
.jn{font:600 .8rem/1 var(--sans);color:var(--straw2);text-align:center}.jrow.first .jn{font-size:.68rem;letter-spacing:.08em;text-transform:uppercase}
.jl{font-size:1rem}.jrow.last .jl{font-weight:600}.worse{font-style:normal;font-size:.72rem;font-weight:700;color:var(--bad);border:1px solid var(--bad);border-radius:99px;padding:2px 8px;margin-left:10px;white-space:nowrap}
.jbar{display:flex;height:14px;border-radius:99px;overflow:hidden;background:var(--line)}.jbar i{width:var(--w);transition:width .9s cubic-bezier(.2,.7,.2,1)}
.js .pre .jbar i{width:0}.s-right{background:var(--good)}.s-partial{background:var(--mid)}.s-wrong{background:var(--bad)}.s-idk{background:var(--idk)}
.jw{text-align:right;font-size:.82rem;color:var(--muted);white-space:nowrap}.jw b{font:600 1.35rem/1 var(--serif);color:var(--bad);margin-right:4px}
.jhead{display:grid;grid-template-columns:44px minmax(0,1.15fr) minmax(0,1fr) 92px;gap:16px;font-size:.75rem;color:var(--muted);letter-spacing:.04em;margin-top:40px}
.jhead span:last-child{text-align:right}.after-note{color:var(--muted);font-size:.92rem;margin-top:20px;max-width:720px}
/* safety net */
.net{background:var(--ink);color:#e9f1ee}.net .lede{color:#b9cbc6}.net h2 em{font-style:italic;color:var(--straw)}
.net-box{margin-top:44px;background:var(--ink2);border:1px solid rgba(233,241,238,.1);border-radius:24px;padding:clamp(22px,4vw,40px)}
.flow{display:grid;grid-template-columns:repeat(4,auto);gap:10px;align-items:center;justify-content:start;font-size:.9rem;color:#b9cbc6;margin-bottom:32px}
.flow span{background:rgba(233,241,238,.06);border:1px solid rgba(233,241,238,.12);border-radius:10px;padding:9px 13px}.flow span.key{border-color:var(--straw);color:var(--straw)}.flow b{color:var(--straw)}
.slider label{display:flex;justify-content:space-between;align-items:baseline;gap:12px;font-size:1.05rem;margin-bottom:12px;flex-wrap:wrap}
.slider output{font:600 1.7rem/1 var(--serif);color:var(--straw)}.slider input{width:100%;accent-color:var(--straw);height:28px}
.ends{display:flex;justify-content:space-between;font-size:.78rem;color:#9fb5af}
.bar100{display:flex;height:46px;border-radius:12px;overflow:hidden;margin:30px 0 18px;background:rgba(233,241,238,.08)}
.bar100 i{transition:width .5s ease;width:var(--w)}.b-right{background:var(--good)}.b-held{background:var(--held)}.b-wrong{background:var(--bad)}.b-idk{background:#56635f}
.of100{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}.of100 b{display:block;font:600 clamp(1.7rem,4vw,2.4rem)/1.1 var(--serif);font-variant-numeric:tabular-nums}
.of100 span{font-size:.85rem;color:#b9cbc6}.of100 div{border-top:3px solid;padding-top:10px}
.of100 .o-right{border-color:var(--good)}.of100 .o-held{border-color:var(--held)}.of100 .o-wrong{border-color:var(--bad)}.of100 .o-idk{border-color:#56635f}
.fine{font-size:.78rem;color:#9fb5af;margin-top:22px}
/* examples */
.chats{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:20px;margin-top:44px}
.chat{background:var(--paper);border:1px solid var(--line);border-radius:20px;padding:20px;display:flex;flex-direction:column}
.chat .bubble{max-width:100%}.chat .bubble p{display:-webkit-box;-webkit-line-clamp:4;-webkit-box-orient:vertical;overflow:hidden}
.kind{align-self:flex-start;font-size:.7rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:var(--straw2);margin-bottom:12px}
.chat details{margin-top:auto;padding-top:6px}details>summary{cursor:pointer;color:var(--teal);font-size:.88rem;font-weight:600}.full{font-size:.86rem;white-space:pre-wrap}
/* how we work */
.how{background:var(--paper)}.steps3{display:grid;grid-template-columns:repeat(3,1fr);gap:clamp(18px,3vw,36px);margin-top:44px;counter-reset:s}
.steps3 div{counter-increment:s;padding-top:20px;border-top:2px solid var(--ink)}.steps3 div::before{content:"0" counter(s);font:600 .85rem/1 var(--sans);color:var(--straw2);display:block;margin-bottom:14px}
.steps3 h3{font-size:1.5rem;margin-bottom:8px}.steps3 p{margin:0;color:var(--muted)}
.cta{margin-top:clamp(56px,8vw,88px);background:var(--ink);color:#e9f1ee;border-radius:28px;padding:clamp(32px,6vw,64px);display:grid;grid-template-columns:auto 1fr auto;gap:28px;align-items:center;position:relative;overflow:hidden}
.cta svg{width:72px;height:84px}.cta h2{font-size:clamp(1.6rem,3.5vw,2.3rem)}.cta p{margin:8px 0 0;color:#b9cbc6}
/* go deeper */
.deep h2{font-size:clamp(1.4rem,3vw,1.9rem)}.layer{background:var(--paper);border:1px solid var(--line);border-radius:14px;margin-top:12px}
.deep .layers{margin-top:28px}.layer>summary{padding:18px 20px;font-size:1rem;font-weight:600;color:var(--fg);list-style:none;display:flex;align-items:baseline;gap:12px;flex-wrap:wrap}
.layer>summary::after{content:"+";color:var(--muted);font-size:1.2rem;margin-left:auto}.layer[open]>summary::after{content:"\\2212"}.layer>summary::-webkit-details-marker{display:none}
.layer>div{padding:0 20px 20px}.layer summary span{color:var(--muted);font-weight:400;font-size:.85rem;margin-left:8px}
.table-wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:.82rem}th,td{padding:7px 9px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{font-weight:600;color:var(--muted)}td{font-variant-numeric:tabular-nums}.table-wrap td:nth-child(2){min-width:220px}.ci{display:block;color:var(--muted);font-size:.72rem}
.gates{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}.gate h3{font:600 .95rem/1.3 var(--sans)}.note{color:var(--muted);font-size:.84rem}
.controls{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px;margin-bottom:12px}
.controls label{font-size:.75rem;color:var(--muted);display:flex;flex-direction:column;gap:4px}
.controls select,.controls input{font:inherit;font-size:.85rem;padding:7px 8px;border-radius:6px;border:1px solid var(--line);background:var(--cream);color:var(--fg);min-width:0}
.count{font-size:.85rem;color:var(--muted);margin:0 0 8px}.qlist{display:grid;gap:8px}
.qitem{border:1px solid var(--line);border-radius:8px;background:var(--cream)}.qitem summary{cursor:pointer;padding:10px 12px;display:grid;grid-template-columns:1fr auto;gap:8px;align-items:start;list-style:none;color:var(--fg);font-weight:400}
.qitem summary::-webkit-details-marker{display:none}.qitem[open] summary{border-bottom:1px solid var(--line)}.qitem.hit{outline:2px solid var(--straw2)}
.qtext{font-size:.9rem}.qid{color:var(--muted);font-size:.75rem;margin-right:6px;font-variant-numeric:tabular-nums}.qtext .tag{font-size:.68rem;font-weight:700;color:var(--straw2);text-transform:uppercase;letter-spacing:.06em;margin-right:6px}
.badges{display:flex;gap:4px;white-space:nowrap}.b{font-size:.7rem;font-weight:600;padding:2px 7px;border-radius:99px;border:1px solid}
.b.good{color:var(--good);border-color:var(--good)}.b.bad{color:var(--bad);border-color:var(--bad)}.b.mid{color:#a8771a;border-color:var(--mid)}
.qbody{padding:12px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:16px}.pair p{margin:4px 0 0;font-size:.88rem;white-space:pre-wrap}.pair h4{margin:0;font-size:.75rem;color:var(--muted)}
.why{font-size:.78rem;color:var(--muted);margin-top:6px}.ref{font-size:.82rem;color:var(--muted);margin:10px 0 0}.flag{font-size:.7rem;color:#a8771a;margin-left:6px}
.more{margin-top:10px;font:inherit;font-size:.85rem;padding:7px 14px;border-radius:6px;border:1px solid var(--line);background:var(--paper);color:var(--fg);cursor:pointer}
ul{padding-left:20px}.deep a{color:var(--teal)}
footer{background:var(--ink);color:#9fb5af;font-size:.82rem;padding:28px 0}footer .wrap{display:flex;gap:10px 20px;flex-wrap:wrap;align-items:center}footer a{color:#e9f1ee}
.js .reveal{opacity:0;transform:translateY(24px);transition:opacity .8s ease,transform .8s ease}.js .reveal.in{opacity:1;transform:none}
@media (max-width:860px){.hero-grid,.ba-grid{grid-template-columns:1fr}.phone{margin:0 auto}.stage .bigmark{display:none}.navlinks a.hide-m{display:none}
.jrow,.jhead{grid-template-columns:34px minmax(0,1fr) 74px}.jrow .jbar{grid-column:2/4;grid-row:2}.jhead span:nth-child(3){display:none}
.flow{grid-template-columns:1fr 1fr}.of100{grid-template-columns:1fr 1fr}.steps3{grid-template-columns:1fr}.cta{grid-template-columns:1fr}.pair{grid-template-columns:1fr}
.qitem summary{grid-template-columns:1fr}.badges{flex-wrap:wrap}.hero{min-height:0}}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}.js .anim .phone>*,.js .anim .stamp,.js .reveal{opacity:1;transform:none}.js .anim .stamp{transform:rotate(6deg)}.js .pre .jbar i{width:var(--w)}html{scroll-behavior:auto}}
"""


def main() -> None:
    runs = [(n, json.loads((RESULTS / n / "metrics.json").read_text(encoding="utf-8")), _rows(n))
            for n in ORDER if (RESULTS / n / "metrics.json").exists()]
    first, last = runs[0], runs[-1]
    m0, m1 = first[1], last[1]
    w0, w1 = wrong_shown(first[2]), wrong_shown(last[2])
    n_q = len(first[2])
    hb, ha = ({r["id"]: r for r in rows}[HERO_ID] for rows in (first[2], last[2]))
    gate = gate_grid(last[0])
    data = explorer_data(runs)
    mark = lambda k: MARK.replace("{k}", k)
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Babui Labs · Stop your chatbot making things up</title>
<meta name="description" content="A company-handbook chatbot gave a confident wrong answer to 1 in 5 questions. After a measured audit and repair: 1 in {round(1 / w1)}.">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,500;0,9..144,600;1,9..144,500&family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
<script>document.documentElement.classList.add("js")</script><style>{CSS}</style></head><body>

<header class="hero anim">
<div class="wrap" style="display:flex;flex-direction:column;flex:1;width:100%">
<nav><a class="brand" href="{CONTACT}">{mark("a")}Babui Labs</a>
<div class="navlinks"><a class="hide-m" href="#results">Results</a><a class="hide-m" href="#how">How we work</a><a href="{REPO}">Code</a></div></nav>
<div class="hero-grid">
<div>
<p class="kicker">RAG audit · a real case study</p>
<h1>Your chatbot sounds sure.<br>Is it <em>right?</em></h1>
<p class="lede">We tested a company-handbook assistant on {n_q} real questions. It confidently made up or got wrong 1 answer in 5. We found out why and fixed it.</p>
<div class="proof"><b class="from">1 in {round(1 / w0)}</b><span class="arrow">→</span><b class="to">1 in {round(1 / w1)}</b>
<span class="cap">confident wrong answers, before and after</span></div>
<div class="ctas"><a class="btn primary" href="#contact">Get your chatbot checked</a><a class="btn ghost" href="#results">See the results ↓</a></div>
</div>
<div class="stage">{mark("b").replace("<svg", '<svg class="bigmark"', 1)}
<div class="phone">
<header><i></i>HR assistant<small>a real answer from the test</small></header>
{_bubble("Employee", "q", html.escape(hb["question"]))}
<div class="bubble bad"><span class="stamp">Made up</span><span class="who">Before</span><p>{html.escape(_first_line(hb["answer"]))}</p>
<span class="verdict">The handbook never says this</span></div>
<div class="divider">after the fix</div>
{_bubble("After", "good", html.escape(_first_line(ha["answer"])), "Honest. Nothing invented.")}
</div></div>
</div>
<div class="trust"><span>{n_q} test questions, written from the documents</span><span>Every answer published</span><span>One change at a time, each one measured</span></div>
</div></header>

<section class="ba" id="results"><div class="wrap">
<p class="kicker reveal">The whole test at a glance</p>
<h2 class="reveal">Same {n_q} questions. Before and after.</h2>
<p class="lede reveal">Each dot is one question. Switch between the chatbot as most tutorials build it and the version after our fixes.</p>
<div class="ba-grid">
<div class="reveal"><div class="toggle" role="group" aria-label="Which chatbot"><button type="button" id="show-before">Before</button><button type="button" id="show-after">After our fixes</button></div>
<div class="dots" id="dots" role="img" aria-label="{n_q} test questions, coloured by how the chatbot answered"></div>
<div class="legend"><span><i style="background:var(--good)"></i>right</span><span><i style="background:var(--mid)"></i>partly right</span>
<span><i style="background:var(--bad)"></i>wrong or made up</span><span><i style="background:var(--idk)"></i>said “I don’t know” but could have answered</span></div></div>
<div class="tally reveal">
<div class="t-right"><b id="t-right">0</b><span>answered right</span><small>including saying “I don’t know” when the handbook has no answer</small></div>
<div class="t-wrong"><b id="t-wrong">0</b><span>wrong or made up</span><small>the answers that cost you customers</small></div>
<div class="t-idk"><b id="t-idk">0</b><span>gave up too early</span><small>the answer was in the documents</small></div>
<p class="hint">Click any dot to read that question and both answers.</p>
</div></div></div></section>

<section><div class="wrap">
<p class="kicker reveal">What we changed</p>
<h2 class="reveal">{WORDS.get(len(runs) - 1, len(runs) - 1)} small changes. Each one measured.</h2>
<p class="lede reveal">We changed one thing at a time and checked the result before moving on. When an idea made things worse, we say so.</p>
<div class="jhead" aria-hidden="true"><span></span><span>Change</span><span>How the {n_q} answers split</span><span>Wrong answers</span></div>
<ol class="journey pre" id="journey">{_journey(runs)}</ol>
<div class="legend"><span><i style="background:var(--good)"></i>right</span><span><i style="background:var(--mid)"></i>partly right</span>
<span><i style="background:var(--idk)"></i>gave up too early</span><span><i style="background:var(--bad)"></i>wrong or made up</span></div>
<p class="after-note reveal">The biggest single win was teaching it to cite the handbook and to say “I don’t know”. The last two steps get more questions right; a few more errors slip through, which is what the safety net below is for.</p>
</div></section>

<section class="net"><div class="wrap">
<p class="kicker reveal">Optional safety net</p>
<h2 class="reveal">Doubtful answers go to a person, <em>not</em> your customer.</h2>
<p class="lede reveal">A separate checker reads every answer against the handbook before it is shown. You decide how strict it is.</p>
<div class="net-box reveal">
<div class="flow"><span>Question</span><span>Chatbot answers</span><span class="key">Checker reads the sources</span><span>Shown, or sent to your team</span></div>
<div class="slider"><label for="alpha">Allow at most this many wrong answers <output id="alpha-out">3%</output></label>
<input type="range" id="alpha" min="0.02" max="0.08" step="0.005" value="0.03"><div class="ends"><span>stricter: more go to your team</span><span>looser: more answered directly</span></div></div>
<div class="bar100" id="bar100"><i class="b-right"></i><i class="b-wrong"></i><i class="b-held"></i><i class="b-idk"></i></div>
<p style="margin:0 0 14px">Out of every 100 questions:</p>
<div class="of100"><div class="o-right"><b id="o-right"></b><span>answered correctly</span></div><div class="o-wrong"><b id="o-wrong"></b><span>shown, not fully right</span></div>
<div class="o-held"><b id="o-held"></b><span>sent to your team</span></div><div class="o-idk"><b id="o-idk"></b><span>“I don’t know”</span></div></div>
<p class="fine" id="g-base"></p>
</div></div></section>

<section><div class="wrap">
<p class="kicker reveal">Real answers</p>
<h2 class="reveal">What the difference looks like</h2>
<p class="lede reveal">Questions the original chatbot got wrong and the fixed one got right, word for word.</p>
<div class="chats reveal">{_examples(first[2], last[2])}</div>
</div></section>

<section class="how" id="how"><div class="wrap">
<p class="kicker reveal">How we work with you</p>
<h2 class="reveal">The same process, on your chatbot.</h2>
<div class="steps3 reveal">
<div><h3>Measure</h3><p>We write a test set from your own documents and score the chatbot you have today, so you know exactly where it fails.</p></div>
<div><h3>Fix</h3><p>Search, document splitting, prompts: one change at a time. We keep only what moves the number.</p></div>
<div><h3>Prove</h3><p>You get a report like this one, every answer included, plus an optional safety net for what remains.</p></div>
</div>
<div class="cta reveal" id="contact">{mark("c")}<div><h2>Is your chatbot making things up?</h2><p>Let’s measure it. You’ll know where it stands before you spend anything on fixes.</p></div>
<a class="btn primary" href="{CONTACT}">Talk to Babui Labs</a></div>
</div></section>

<section class="deep"><div class="wrap">
<h2>Go deeper</h2>
<p class="lede">Everything behind the numbers, for whoever wants to check.</p>
<div class="layers">
<details class="layer" id="explorer"><summary>Check every answer yourself<span>{n_q} questions, any two steps side by side</span></summary><div>
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
<p class="note">95% bootstrap intervals under each value. Differences inside overlapping intervals are not established improvements.
Versions: {", ".join(f"<code>{n.split('_')[0]}</code> {html.escape(PLAIN.get(n, n).lower())}" for n, _, _ in runs)}.</p>
{_table(runs)}</div></details>

<details class="layer"><summary>The safety net in detail<span>method and published targets</span></summary><div>
<p class="note">The checker is a third model family (neither the assistant nor the grader). Its threshold is set with conformal risk control on
half the questions and tested on the other half, averaged over 1000 random splits. The guarantee holds on average, not on every split;
with 52 calibration questions single splits vary, as the 90th-percentile column shows.</p>
<div class="gates">{_gate()}</div></div></details>

<details class="layer"><summary>How it was measured, and the limits<span>read before trusting the numbers</span></summary><div><ul>
<li>{n_q} questions over 94 pages of the public GitLab Handbook: facts, paraphrases, numbers, cross-country comparisons, and 23 questions the handbook cannot answer. 12 unsound drafts were removed, with reasons published.</li>
<li>“Wrong answers” counts answers shown to the user that are wrong or invented; a partly right answer counts as half.</li>
<li>Assistant <code>{m1["generator"]}</code>; grader <code>{m1["judge"]}</code>, a different model family. The grader was checked against blind grades from an independent LLM reviewer of a third family (agreement κ = 0.93, n = 42). That is a cross-check between models, not a human gold standard.</li>
<li>Only 14 comparison questions, so that number has a wide interval ({_pct(stat(m0, "multi_hop")["value"])} → {_pct(stat(m1, "multi_hop")["value"])}).</li>
<li>One corpus, one assistant model. Your system will behave differently, which is the reason to measure it.</li>
<li>Every model response is committed, so all numbers replay offline without an API key.</li>
</ul></div></details>
</div></div></section>

<footer><div class="wrap"><span>RAG Eval Kit by Babui Labs</span><a href="{REPO}">Code</a>
<a href="{REPO}/blob/main/docs/CASE_STUDY.md">Case study</a>
<span>Corpus: GitLab Handbook, CC BY-SA 4.0, © GitLab B.V. · Code: MIT</span></div></footer>
{_json_script("gate-data", gate)}
{_json_script("explorer-data", data)}
<script>{JS.read_text(encoding="utf-8")}</script>
</body></html>
"""
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(page, encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
