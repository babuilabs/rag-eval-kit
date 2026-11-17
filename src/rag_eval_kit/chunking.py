"""Two chunking strategies: fixed-size windows (the common default) and heading-aware sections."""
from __future__ import annotations

import re
from dataclasses import dataclass

from .corpus import Document

HEADING = re.compile(r"^(#{1,4})\s+(.*)$", re.M)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str


def fixed_chunks(doc: Document, size: int = 1000, overlap: int = 0) -> list[Chunk]:
    """Slice the raw text every `size` characters, ignoring structure."""
    step = max(1, size - overlap)
    out = []
    for i, start in enumerate(range(0, len(doc.text), step)):
        piece = doc.text[start:start + size].strip()
        if piece:
            out.append(Chunk(f"{doc.doc_id}#f{i}", doc.doc_id, doc.title, piece))
    return out


def _sections(doc: Document) -> list[tuple[str, str]]:
    """Split on markdown headings; return (heading path, body) pairs."""
    path: list[str] = []
    out: list[tuple[str, str]] = []
    last, last_path = 0, [doc.title]
    for m in HEADING.finditer(doc.text):
        body = doc.text[last:m.start()].strip()
        if body:
            out.append((" > ".join(last_path), body))
        level = len(m.group(1))
        path = path[:level - 1] + [m.group(2).strip()]
        last, last_path = m.end(), [doc.title] + path
    body = doc.text[last:].strip()
    if body:
        out.append((" > ".join(last_path), body))
    return out


def _split_long(body: str, max_chars: int) -> list[str]:
    if len(body) <= max_chars:
        return [body]
    parts, cur = [], ""
    for para in re.split(r"\n\s*\n", body):
        if cur and len(cur) + len(para) + 2 > max_chars:
            parts.append(cur)
            cur = para
        else:
            cur = f"{cur}\n\n{para}" if cur else para
    if cur:
        parts.append(cur)
    # a single huge paragraph (e.g. a big table) still gets cut by size
    final = []
    for p in parts:
        if len(p) > max_chars * 1.5:
            final.extend(p[i:i + max_chars] for i in range(0, len(p), max_chars))
        else:
            final.append(p)
    return final


def heading_chunks(doc: Document, max_chars: int = 1500, min_chars: int = 200) -> list[Chunk]:
    """One chunk per section, prefixed with its heading path; tiny sections merge forward."""
    out: list[Chunk] = []
    pending_head, pending = "", ""
    for head, body in _sections(doc):
        if pending:
            body = f"{pending}\n\n{body}"
            head = pending_head
            pending = ""
        if len(body) < min_chars:
            pending_head, pending = head, body
            continue
        for piece in _split_long(body, max_chars):
            out.append(Chunk(f"{doc.doc_id}#h{len(out)}", doc.doc_id, doc.title, f"[{head}]\n{piece}"))
    if pending:
        out.append(Chunk(f"{doc.doc_id}#h{len(out)}", doc.doc_id, doc.title, f"[{pending_head}]\n{pending}"))
    return out


def chunk_corpus(docs: list[Document], strategy: str, **kw) -> list[Chunk]:
    fn = {"fixed": fixed_chunks, "heading": heading_chunks}[strategy]
    return [c for d in docs for c in fn(d, **kw)]
