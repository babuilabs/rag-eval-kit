"""Load the markdown corpus and normalise text for matching."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.S)
SHORTCODE = re.compile(r"\{\{[<%].*?[>%]\}\}", re.S)
HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)
HTML_TAG = re.compile(r"</?[a-zA-Z][^>]*>")
TITLE = re.compile(r"^title:\s*[\"']?(.*?)[\"']?\s*$", re.M)


@dataclass
class Document:
    doc_id: str   # path relative to the corpus root, e.g. "total-rewards/benefits/_index.md"
    title: str
    text: str     # cleaned markdown


def clean_markdown(raw: str) -> tuple[str, str]:
    title = ""
    m = FRONT_MATTER.match(raw)
    if m:
        t = TITLE.search(m.group(1))
        title = t.group(1).strip() if t else ""
        raw = raw[m.end():]
    raw = HTML_COMMENT.sub("", raw)
    raw = SHORTCODE.sub("", raw)
    raw = HTML_TAG.sub("", raw)
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    return title, raw.strip()


def load_corpus(root: str | Path) -> list[Document]:
    root = Path(root)
    docs = []
    for p in sorted(root.rglob("*.md")):
        title, text = clean_markdown(p.read_text(encoding="utf-8", errors="replace"))
        doc_id = p.relative_to(root).as_posix()
        if not title:
            title = doc_id.rsplit("/", 1)[0] if doc_id.endswith("_index.md") else doc_id
        if len(text) > 200:
            docs.append(Document(doc_id, title, text))
    return docs


_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_PUNCT = re.compile(r"[*_`#>|\[\]()]+")
_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Lower-case, drop markdown syntax and link targets, collapse whitespace."""
    text = _LINK.sub(r"\1", text)
    text = _PUNCT.sub(" ", text)
    return _WS.sub(" ", text).strip().lower()


def shingles(text: str, n: int = 5) -> set[tuple[str, ...]]:
    w = normalize(text).split()
    if len(w) < n:
        return {tuple(w)} if w else set()
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def evidence_overlap(evidence: str, passage: str, n: int = 5) -> float:
    """Fraction of the evidence's word 5-grams found in the passage (1.0 = fully contained)."""
    e = shingles(evidence, n)
    if not e:
        return 0.0
    return len(e & shingles(passage, n)) / len(e)
