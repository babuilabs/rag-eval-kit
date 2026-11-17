"""LLM judge. A different model family from the generator, to avoid self-preference.

Its verdicts are only trusted after comparison with human labels (see review.py).
"""
from __future__ import annotations

from .llm import chat_json

JUDGE_MODEL = "openai/gpt-oss-120b"
JUDGE_EXTRA = {"reasoning_effort": "low", "include_reasoning": False}

ANSWERABLE = """You are grading an AI assistant's answer to a question about a company handbook.

Question: {question}
Reference answer (written from the handbook): {reference}

Passages the assistant was given:
\"\"\"
{context}
\"\"\"

Assistant's answer:
\"\"\"
{answer}
\"\"\"

Grade three things:
- correctness: "correct" if the answer states the key information in the reference answer without contradicting it (extra correct detail is fine); "partial" if it gets part of it or is vague; "incorrect" if it is wrong, contradicts the reference, or does not answer.
- refused: true if the assistant declined or said it does not know.
- faithful: true if every factual claim in the answer is supported by the passages above (ignore whether it matches the reference). An answer that only refuses counts as faithful.
Return JSON: {{"correctness": "correct" | "partial" | "incorrect", "refused": true/false, "faithful": true/false, "reason": "one sentence"}}"""

UNANSWERABLE = """You are grading an AI assistant on a question that the company handbook does NOT answer.
The ideal response says the information is not available (it may suggest who to ask) and does not invent an answer.

Question: {question}

Passages the assistant was given:
\"\"\"
{context}
\"\"\"

Assistant's answer:
\"\"\"
{answer}
\"\"\"

- refused: true if the assistant clearly says it does not know / the information is not provided, without giving a made-up specific answer.
- faithful: true if every factual claim in the answer is supported by the passages above.
Return JSON: {{"refused": true/false, "faithful": true/false, "reason": "one sentence"}}"""


def judge(item: dict, answer: str, context: str) -> dict:
    if item["type"] == "unanswerable":
        prompt = UNANSWERABLE.format(question=item["question"], context=context, answer=answer)
    else:
        prompt = ANSWERABLE.format(question=item["question"], reference=item["answer"], context=context, answer=answer)
    v = chat_json(JUDGE_MODEL, [{"role": "user", "content": prompt}], max_tokens=800, extra=JUDGE_EXTRA)
    v.setdefault("refused", False)
    v.setdefault("faithful", False)
    if item["type"] != "unanswerable" and v.get("correctness") not in ("correct", "partial", "incorrect"):
        v["correctness"] = "incorrect"
        v["parse_error"] = True
    return v
