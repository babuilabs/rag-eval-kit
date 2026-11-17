"""The RAG pipeline under test: retrieve k chunks, then answer with an LLM."""
from __future__ import annotations

from .llm import chat

GEN_MODEL = "qwen/qwen3.8-27b"
GEN_EXTRA = {"reasoning_effort": "none"}

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
