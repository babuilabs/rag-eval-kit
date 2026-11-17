"""The RAG pipeline under test: retrieve k chunks, then answer with an LLM."""
from __future__ import annotations

from .llm import chat, chat_json

GEN_MODEL = "qwen/qwen3.8-27b"
GEN_EXTRA = {"reasoning_effort": "none"}

# Comparison questions ("UK vs Ireland pension") need passages about each side, but one
# search tends to fill the top-k with whichever side matches best. Splitting the question
# into one search per side fixes that; simple questions come back unchanged.
DECOMPOSE = """Split this question about a company handbook into the separate searches needed to answer it.
- If it compares or combines several countries, entities, policies or topics, write one short search per item.
- If one search is enough, return the question unchanged as the only search.
- At most 3 searches.
Question: {question}
Return JSON: {{"searches": ["...", "..."]}}"""


def decompose(question: str) -> list[str]:
    out = chat_json(GEN_MODEL, [{"role": "user", "content": DECOMPOSE.format(question=question)}],
                    max_tokens=200, extra=GEN_EXTRA)
    qs = [q.strip() for q in out.get("searches", []) if isinstance(q, str) and q.strip()][:3]
    return qs or [question]


def merge_hits(rankings: list[list[tuple[int, float]]], k: int) -> list[tuple[int, float]]:
    """Take results round-robin across searches (best of each first), without duplicates."""
    seen, merged = set(), []
    for rank in range(max((len(r) for r in rankings), default=0)):
        for r in rankings:
            if rank < len(r) and r[rank][0] not in seen:
                seen.add(r[rank][0])
                merged.append(r[rank])
    return merged[:k]

# What most tutorials ship: stuff the context in, ask the question.
BASIC = """Use the following context to answer the question.

Context:
{context}

Question: {question}
Answer:"""

# Grounded: answer only from the context, cite passages, admit when the answer is missing.
GROUNDED_SYSTEM = """You answer questions about the GitLab company handbook using ONLY the numbered passages provided.
- Base every statement on the passages and cite them like [1] or [2][3].
- If the passages do not contain the answer, reply exactly: "I don't know based on the handbook." Do not guess or use outside knowledge.
- If the answer depends on a country, entity or situation, say which one the passage covers.
- Be concise: at most 3 sentences."""

GROUNDED_USER = """Passages:
{context}

Question: {question}"""


def format_context(chunks: list[str], numbered: bool) -> str:
    if numbered:
        return "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(chunks))
    return "\n\n".join(chunks)


def answer(question: str, chunks: list[str], prompt: str) -> dict:
    if prompt == "basic":
        msgs = [{"role": "user", "content": BASIC.format(context=format_context(chunks, False), question=question)}]
    elif prompt == "grounded":
        msgs = [{"role": "system", "content": GROUNDED_SYSTEM},
                {"role": "user", "content": GROUNDED_USER.format(context=format_context(chunks, True), question=question)}]
    else:
        raise ValueError(prompt)
    out = chat(GEN_MODEL, msgs, max_tokens=400, extra=GEN_EXTRA)
    return {"answer": out["content"].strip(), "latency_s": out["latency_s"], "usage": out["usage"]}
