"""Build the evaluation set: draft questions with one model, check them with another.

Pipeline
  1. sample source sections from the corpus (seeded, spread across pages)
  2. draft a question + reference answer + verbatim evidence (drafting model)
  3. automatic filters: evidence must really appear in the source, no near-duplicates
  4. independent check by a stronger model: self-contained? answer correct and supported?
  5. unanswerable questions: kept only if a full-corpus search finds no passage that answers them

Every draft, verdict and rejection reason is written to results/evalset_build/ for audit.
A human then reviews a sample (see review sheet) before any result is trusted.

    python -m rag_eval_kit.evalset build
"""
from __future__ import annotations

import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .chunking import heading_chunks
from .corpus import Document, evidence_overlap, load_corpus, normalize
from .llm import DailyLimitReached, chat_json

DRAFT_MODEL = "openai/gpt-oss-20b"
CHECK_MODEL = "openai/gpt-oss-120b"
REASONING = {"reasoning_effort": "low", "include_reasoning": False}
SEED = 42
PLAN = {"fact": 34, "numeric": 33, "paraphrase": 33, "multi_hop": 30, "unanswerable": 40}
TARGET = {"single": 75, "multi_hop": 20, "unanswerable": 25}
OUT = Path("data/evalset.jsonl")
AUDIT = Path("results/evalset_build")

TYPE_RULE = {
    "fact": "The answer is a specific fact stated in the excerpt.",
    "numeric": "The answer is a number, amount, percentage, duration or date stated in the excerpt.",
    "paraphrase": "Word the question differently from the excerpt: do not copy any phrase of 4+ words from it.",
}

DRAFT_SINGLE = """You are writing test questions for an assistant that answers questions about the GitLab company handbook.

Handbook page: "{title}"
Section: {head}
Excerpt:
\"\"\"
{text}
\"\"\"

Write ONE question a GitLab team member might realistically ask that is fully answerable from this excerpt alone.
Rules:
- Self-contained: name the specific country, entity, benefit or policy it is about. Never say "this section", "the excerpt" or "the document".
- {type_rule}
- The answer is short (at most 2 sentences) and correct according to the excerpt.
- "evidence" is copied VERBATIM from the excerpt: the 1-3 consecutive sentences that contain the answer.
If the excerpt has no clear fact worth asking about (e.g. it is only links or a table of contents), return {{"skip": true}}.
Return JSON: {{"question": "...", "answer": "...", "evidence": "..."}}"""

DRAFT_MULTI = """You are writing test questions for an assistant that answers questions about the GitLab company handbook.

Excerpt A, from "{title_a}" ({head_a}):
\"\"\"
{text_a}
\"\"\"

Excerpt B, from "{title_b}" ({head_b}):
\"\"\"
{text_b}
\"\"\"

Write ONE realistic question whose correct answer REQUIRES information from BOTH excerpts (for example a comparison, or a combination of two facts).
Rules:
- Self-contained: name the countries, entities or policies explicitly. Never mention "excerpt A/B".
- The answer is short (at most 2 sentences) and correct according to the excerpts.
- "evidence_a" and "evidence_b" are copied VERBATIM: the sentence(s) from each excerpt that the answer depends on.
If no sensible question needs both excerpts, return {{"skip": true}}.
Return JSON: {{"question": "...", "answer": "...", "evidence_a": "...", "evidence_b": "..."}}"""

DRAFT_UNANSWERABLE = """You are writing test questions for an assistant that answers questions about the GitLab company handbook.
A good assistant must say it does not know when the handbook does not contain the answer.

Handbook page: "{title}"
Section: {head}
Excerpt:
\"\"\"
{text}
\"\"\"

Write ONE realistic question on the same topic that a team member might ask, but which this excerpt does NOT answer and which a company handbook would plausibly not cover (a specific detail, scenario or exception that is not stated). It must not be answerable by common sense alone.
Self-contained: name the specific country, benefit or policy. Never say "this section" or "the excerpt".
Return JSON: {{"question": "...", "why_not_answerable": "..."}}"""

CHECK_ANSWERABLE = """You are auditing a test question for a handbook question-answering benchmark.

Source text:
\"\"\"
{source}
\"\"\"

Question: {question}
Reference answer: {answer}

Check all of the following:
1. self_contained: the question makes sense on its own and is not ambiguous (it names the country/entity/policy where that matters).
2. supported: the reference answer is correct and fully supported by the source text.
3. specific: the question has one clear correct answer (it would not get a different answer for a different country, team or time period that it fails to specify).
Return JSON: {{"self_contained": true/false, "supported": true/false, "specific": true/false, "reason": "one sentence"}}"""

CHECK_UNANSWERABLE = """You are auditing a test question that is supposed to be UNANSWERABLE from a company handbook.

Here are the handbook passages most related to the question:
{passages}

Question: {question}

Can the question be answered (fully or substantially) from these passages?
Return JSON: {{"answerable": "yes" | "partially" | "no", "reason": "one sentence"}}"""


def _llm(model: str, prompt: str, max_tokens: int = 1200) -> dict:
    return chat_json(model, [{"role": "user", "content": prompt}], max_tokens=max_tokens, extra=REASONING)


def _usable(text: str) -> bool:
    letters = sum(ch.isalpha() for ch in text)
    links = text.count("](")
    return 400 <= len(text) <= 2500 and letters / max(1, len(text)) > 0.6 and links < 8


def sample_sections(docs: list[Document], rng: random.Random):
    sections = []
    for d in docs:
        for c in heading_chunks(d, max_chars=2500):
            body = c.text.split("\n", 1)[1] if c.text.startswith("[") else c.text
            head = c.text[1:c.text.index("]")] if c.text.startswith("[") else d.title
            if _usable(body):
                sections.append({"doc_id": d.doc_id, "title": d.title, "head": head, "text": body})
    rng.shuffle(sections)
    # spread across pages: round-robin by document
    by_doc: dict[str, list] = {}
    for s in sections:
        by_doc.setdefault(s["doc_id"], []).append(s)
    spread = []
    while any(by_doc.values()):
        for k in list(by_doc):
            if by_doc[k]:
                spread.append(by_doc[k].pop())
    return spread


def draft_single(sec, qtype):
    r = _llm(DRAFT_MODEL, DRAFT_SINGLE.format(type_rule=TYPE_RULE[qtype], **sec))
    if r.get("skip") or not r.get("question"):
        return None, "draft skipped"
    ev = r.get("evidence", "")
    if evidence_overlap(ev, sec["text"]) < 0.8:
        return None, "evidence not verbatim"
    item = {"type": qtype, "question": r["question"].strip(), "answer": r.get("answer", "").strip(),
            "evidence": [{"doc_id": sec["doc_id"], "quote": ev.strip()}], "source": [sec]}
    v = _llm(CHECK_MODEL, CHECK_ANSWERABLE.format(source=sec["text"], question=item["question"], answer=item["answer"]), max_tokens=800)
    item["check"] = v
    if not (v.get("self_contained") and v.get("supported") and v.get("specific")):
        return None, f"check failed: {v.get('reason', v)}"
    return item, None


def draft_multi(pair):
    a, b = pair
    r = _llm(DRAFT_MODEL, DRAFT_MULTI.format(
        title_a=a["title"], head_a=a["head"], text_a=a["text"],
        title_b=b["title"], head_b=b["head"], text_b=b["text"]))
    if r.get("skip") or not r.get("question"):
        return None, "draft skipped"
    if evidence_overlap(r.get("evidence_a", ""), a["text"]) < 0.8 or evidence_overlap(r.get("evidence_b", ""), b["text"]) < 0.8:
        return None, "evidence not verbatim"
    item = {"type": "multi_hop", "question": r["question"].strip(), "answer": r.get("answer", "").strip(),
            "evidence": [{"doc_id": a["doc_id"], "quote": r["evidence_a"].strip()},
                         {"doc_id": b["doc_id"], "quote": r["evidence_b"].strip()}], "source": [a, b]}
    source = f"[{a['title']} > {a['head']}]\n{a['text']}\n\n[{b['title']} > {b['head']}]\n{b['text']}"
    v = _llm(CHECK_MODEL, CHECK_ANSWERABLE.format(source=source, question=item["question"], answer=item["answer"]), max_tokens=800)
    item["check"] = v
    if not (v.get("self_contained") and v.get("supported") and v.get("specific")):
        return None, f"check failed: {v.get('reason', v)}"
    return item, None


def draft_unanswerable(sec, search):
    r = _llm(DRAFT_MODEL, DRAFT_UNANSWERABLE.format(**sec))
    if not r.get("question"):
        return None, "draft skipped"
    q = r["question"].strip()
    passages = "\n\n".join(f"[{i + 1}] {t}" for i, t in enumerate(search(q)))
    v = _llm(CHECK_MODEL, CHECK_UNANSWERABLE.format(passages=passages, question=q), max_tokens=800)
    item = {"type": "unanswerable", "question": q, "answer": "", "evidence": [],
            "why_not_answerable": r.get("why_not_answerable", ""), "source": [sec], "check": v}
    if v.get("answerable") != "no":
        return None, f"corpus can answer it ({v.get('answerable')}): {v.get('reason', '')}"
    return item, None


def _dedupe(items):
    kept, seen = [], []
    for it in items:
        toks = set(normalize(it["question"]).split())
        if any(len(toks & s) / max(1, len(toks | s)) > 0.6 for s in seen):
            continue
        seen.append(toks)
        kept.append(it)
    return kept


def build() -> int:
    from .chunking import chunk_corpus
    from .retrieval import build_retriever

    rng = random.Random(SEED)
    docs = load_corpus("data/corpus")
    secs = sample_sections(docs, rng)
    n_single = PLAN["fact"] + PLAN["numeric"] + PLAN["paraphrase"]
    single_jobs = [(secs[i], t) for i, t in enumerate(
        ["fact"] * PLAN["fact"] + ["numeric"] * PLAN["numeric"] + ["paraphrase"] * PLAN["paraphrase"])]
    unans_secs = secs[n_single:n_single + PLAN["unanswerable"]]
    # multi-hop pairs: same topic across two different pages (e.g. two countries' benefits),
    # or two different sections of one page
    rest = secs[n_single + PLAN["unanswerable"]:]
    pairs = []
    for s in rest:
        if len(pairs) >= PLAN["multi_hop"]:
            break
        topic = s["doc_id"].rsplit("/", 1)[0]
        mates = [t for t in rest if t is not s and t["doc_id"].rsplit("/", 1)[0] == topic
                 and t["doc_id"] != s["doc_id"]] or [t for t in rest if t is not s and t["doc_id"] == s["doc_id"]]
        if mates:
            pairs.append((s, rng.choice(mates)))

    chunks = chunk_corpus(docs, "heading")
    retr = build_retriever(chunks, "hybrid", rerank=True)
    search = lambda q: [chunks[i].text for i, _ in retr.search(q, 8)]

    AUDIT.mkdir(parents=True, exist_ok=True)
    kept, rejected = [], []

    def run(job):
        kind, arg = job
        try:
            if kind == "single":
                return kind, *draft_single(*arg)
            if kind == "multi":
                return kind, *draft_multi(arg)
            return kind, *draft_unanswerable(arg, search)
        except DailyLimitReached as e:
            return kind, None, f"DAILY LIMIT: {e}"
        except Exception as e:  # keep going; record why
            return kind, None, f"error: {e}"

    jobs = [("single", j) for j in single_jobs] + [("multi", p) for p in pairs] + [("unans", s) for s in unans_secs]
    rng.shuffle(jobs)  # interleave so both models stay busy
    with ThreadPoolExecutor(max_workers=4) as pool:
        for n, (kind, item, why) in enumerate(pool.map(run, jobs), 1):
            if item:
                kept.append(item)
            else:
                rejected.append({"kind": kind, "reason": why})
            if n % 10 == 0:
                print(f"{n}/{len(jobs)} processed, {len(kept)} kept", flush=True)

    kept = _dedupe(kept)
    singles = [k for k in kept if k["type"] in TYPE_RULE][:TARGET["single"]]
    multis = [k for k in kept if k["type"] == "multi_hop"][:TARGET["multi_hop"]]
    unans = [k for k in kept if k["type"] == "unanswerable"][:TARGET["unanswerable"]]
    final = singles + multis + unans
    for i, it in enumerate(final):
        it["id"] = f"q{i + 1:03d}"
    with OUT.open("w", encoding="utf-8") as f:
        for it in final:
            rec = {k: it[k] for k in ("id", "type", "question", "answer", "evidence")}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    (AUDIT / "kept_full.jsonl").write_text(
        "".join(json.dumps(it, ensure_ascii=False) + "\n" for it in final), encoding="utf-8")
    (AUDIT / "rejected.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rejected), encoding="utf-8")
    summary = {"drafted": len(jobs), "passed_checks": len(kept) , "final": len(final),
               "by_type": {t: sum(1 for x in final if x["type"] == t) for t in sorted({x["type"] for x in final})},
               "rejections": {}}
    for r in rejected:
        key = r["reason"].split(":")[0]
        summary["rejections"][key] = summary["rejections"].get(key, 0) + 1
    (AUDIT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


TOPICS = {
    "parental leave": r"parental|maternity|paternity",
    "pension or retirement": r"pension|retirement|401|superannuation",
    "medical insurance": r"medical|health insurance|healthcare",
    "sick leave": r"sick",
    "life insurance": r"life insurance|life assurance",
    "public holidays and vacation": r"holiday|vacation|annual leave",
}

CHECK_NATURAL = CHECK_ANSWERABLE.replace(
    "Return JSON:",
    "4. natural: a real employee would plausibly ask exactly this question (not two unrelated questions glued together).\nReturn JSON:"
).replace('"specific": true/false,', '"specific": true/false, "natural": true/false,')


def comparison_pairs(docs: list[Document], rng: random.Random, n: int):
    """Same benefit topic in two different countries' benefit pages: natural comparison questions."""
    pages = [d for d in docs if "general-and-entity-benefits/" in d.doc_id and not d.doc_id.endswith("_index.md")]
    by_topic: dict[str, list] = {}
    for d in pages:
        for c in heading_chunks(d, max_chars=2500):
            if not c.text.startswith("["):
                continue
            head = c.text[1:c.text.index("]")]
            body = c.text.split("\n", 1)[1]
            for topic, pat in TOPICS.items():
                if re.search(pat, head, re.I) and 300 <= len(body) <= 2500:
                    by_topic.setdefault(topic, []).append(
                        {"doc_id": d.doc_id, "title": d.title, "head": head, "text": body})
    pairs, seen = [], set()
    topics = sorted(by_topic)
    for _ in range(n * 20):
        if len(pairs) >= n:
            break
        t = rng.choice(topics)
        a, b = rng.sample(by_topic[t], 2) if len(by_topic[t]) > 1 else (None, None)
        if not a or a["doc_id"] == b["doc_id"]:
            continue
        key = tuple(sorted((a["doc_id"], b["doc_id"]))) + (t,)
        if key not in seen:
            seen.add(key)
            pairs.append((a, b))
    return pairs


def draft_comparison(pair):
    a, b = pair
    r = _llm(DRAFT_MODEL, DRAFT_MULTI.format(
        title_a=a["title"], head_a=a["head"], text_a=a["text"],
        title_b=b["title"], head_b=b["head"], text_b=b["text"]).replace(
        "(for example a comparison, or a combination of two facts)",
        "(a comparison between the two countries or entities, as an employee deciding between them or moving between them would ask)"))
    if r.get("skip") or not r.get("question"):
        return None, "draft skipped"
    if evidence_overlap(r.get("evidence_a", ""), a["text"]) < 0.8 or evidence_overlap(r.get("evidence_b", ""), b["text"]) < 0.8:
        return None, "evidence not verbatim"
    item = {"type": "multi_hop", "question": r["question"].strip(), "answer": r.get("answer", "").strip(),
            "evidence": [{"doc_id": a["doc_id"], "quote": r["evidence_a"].strip()},
                         {"doc_id": b["doc_id"], "quote": r["evidence_b"].strip()}], "source": [a, b]}
    source = f"[{a['title']} > {a['head']}]\n{a['text']}\n\n[{b['title']} > {b['head']}]\n{b['text']}"
    v = _llm(CHECK_MODEL, CHECK_NATURAL.format(source=source, question=item["question"], answer=item["answer"]), max_tokens=800)
    item["check"] = v
    if not all(v.get(k) for k in ("self_contained", "supported", "specific", "natural")):
        return None, f"check failed: {v.get('reason', v)}"
    return item, None


def rebuild_multihop(n_drafts: int = 32) -> int:
    """Replace the multi-hop items with cross-country comparison questions."""
    rng = random.Random(SEED + 1)
    docs = load_corpus("data/corpus")
    pairs = comparison_pairs(docs, rng, n_drafts)
    new, rejected = [], []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for item, why in pool.map(lambda p: _safe(draft_comparison, p), pairs):
            (new.append(item) if item else rejected.append({"kind": "multi_v2", "reason": why}))
    new = _dedupe(new)[:TARGET["multi_hop"]]
    full = [json.loads(l) for l in (AUDIT / "kept_full.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    old_multi = [x for x in full if x["type"] == "multi_hop"]
    keep = [x for x in full if x["type"] != "multi_hop"]
    final = [x for x in keep if x["type"] != "unanswerable"] + new + [x for x in keep if x["type"] == "unanswerable"]
    for i, it in enumerate(final):
        it["id"] = f"q{i + 1:03d}"
    with OUT.open("w", encoding="utf-8") as f:
        for it in final:
            f.write(json.dumps({k: it[k] for k in ("id", "type", "question", "answer", "evidence")}, ensure_ascii=False) + "\n")
    (AUDIT / "kept_full.jsonl").write_text("".join(json.dumps(it, ensure_ascii=False) + "\n" for it in final), encoding="utf-8")
    (AUDIT / "multihop_v1_replaced.jsonl").write_text(
        "".join(json.dumps(it, ensure_ascii=False) + "\n" for it in old_multi), encoding="utf-8")
    with (AUDIT / "rejected.jsonl").open("a", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rejected)
    s = json.loads((AUDIT / "summary.json").read_text())
    s["multihop_v2"] = {"pairs_drafted": len(pairs), "kept": len(new),
                        "note": "v1 multi-hop items paired unrelated sections and read unnaturally; replaced with cross-country comparisons checked for naturalness"}
    s["final"] = len(final)
    s["by_type"] = {t: sum(1 for x in final if x["type"] == t) for t in sorted({x["type"] for x in final})}
    (AUDIT / "summary.json").write_text(json.dumps(s, indent=2), encoding="utf-8")
    print(json.dumps(s["multihop_v2"] | {"final": len(final), "by_type": s["by_type"]}, indent=2))
    return 0


def _safe(fn, arg):
    try:
        return fn(arg)
    except Exception as e:
        return None, f"error: {e}"


if __name__ == "__main__":
    if sys.argv[1:] == ["build"]:
        sys.exit(build())
    if sys.argv[1:] == ["multihop"]:
        sys.exit(rebuild_multihop())
    sys.exit(__doc__)
