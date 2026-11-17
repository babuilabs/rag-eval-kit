# RAG Eval Kit

Measure why a RAG assistant gives wrong answers, fix it one change at a time, and show the
number each change moved, including the changes that made things worse.

**[Interactive report](https://babuilabs.github.io/rag-eval-kit/)** · [Case study](docs/CASE_STUDY.md) · [Full results table](results/COMPARISON.md)

## Results

A typical tutorial-style RAG assistant over a company handbook (94 pages of the public GitLab
Handbook), scored on 104 questions, 23 of which the handbook cannot answer.

| | Right answers delivered | Wrong answers shown | Says "don't know" when it should | Comparison questions (n=14) |
|---|---|---|---|---|
| **v0** tutorial baseline | 52% | **20.7%** | 83% | 25% |
| **v4** + better chunks, reranker, grounded prompt | 63% | **2.4%** | 100% | 25% |
| **v7** + more passages, one search per compared item | **68%** | 3.4% | 100% | **64%** |
| **v4 + safe gate** (target α = 0.03) | 56% | **1.7%** | 100% | – |

*Right answers delivered* and *wrong answers shown* are shares of all 104 questions (the best
possible is 78% right, since 23 have no answer). A partial answer counts half right, half wrong.
Answering an unanswerable question counts as wrong.

**There is no single "best" version; there is a trade-off you choose.** The baseline showed a wrong
answer to one question in five. A grounded prompt cut that to 1 in 40 but made the assistant refuse
16% of questions it could have answered. Retrieving more, and searching each side of a comparison
separately, won most of those back (comparison questions 25% → 64%) at the cost of one more wrong
answer per 100 questions. A verifier gate on top of v4 pushes wrong answers below 2% by routing more
questions to a person.

All numbers carry 95% bootstrap confidence intervals in the [full table](results/COMPARISON.md) and
the [report](https://babuilabs.github.io/rag-eval-kit/). With 104 questions, gaps of a few points
between neighbouring versions are inside those intervals and should not be read as improvements.

## What each change did

| Version | Change | Answer accuracy | Evidence recall@k | Wrong shown |
|---|---|---|---|---|
| v0 | Fixed 1000-character chunks, dense search, top 3, plain prompt | 67% | 61% | 20.7% |
| v1 | Chunks follow the document's headings | 82% | 86% | 10.1% |
| v2 | Hybrid search (BM25 + dense) | 78% | 86% | 12.0% |
| v3 | Cross-encoder reranker | 84% | 89% | 9.6% |
| v4 | Grounded prompt: cite passages, say "I don't know" | 81% | 89% | 2.4% |
| v5 | Safe gate (a filter on v4/v7, see below) | – | – | 1.7% at α=0.03 |
| v6 | Six passages instead of three | 85% | 91% | 4.3% |
| v7 | An LLM splits comparison questions into one search per item | 87% | 93% | 3.4% |

- **Chunking was the biggest single fix.** Splitting on headings instead of every 1000 characters
  raised evidence recall from 61% to 86%.
- **Hybrid search did not help here.** Adding BM25 left recall unchanged and lowered both accuracy
  (82% → 78%) and correct refusals (83% → 74%), though the gaps are inside the confidence
  intervals. It stays in the table because knowing what not to ship is part of an audit.
- **The prompt mattered most for wrong answers.** v4 never answered an unanswerable question, but
  refused 13 answerable ones. In 12 of those 13 the right passage had not been retrieved, so the
  refusal was correct behaviour and the remaining problem was retrieval, which v6 and v7 address.
- **v7's searches are rewritten as well as split.** The model rewrote 59 of the 67 questions it did
  not split into keyword-style queries, so v7 measures decomposition and query rewriting together.
  It also split two simple questions it should not have, and both turned into refusals.

## Safe gate (v5)

Before an answer is shown, a verifier from a third model family (`google/gemma-3-27b-it`, neither the
generator nor the judge) estimates the probability that every claim is supported by the retrieved
passages. The threshold is set with
[conformal risk control](https://arxiv.org/abs/2208.02814) so that the *expected* share of wrong
answers shown stays below a target α. Answers below the threshold would go to a person.

| Source | Verifier AUROC | No gate (wrong / answered) | α = 0.03 | α = 0.05 | α = 0.10 |
|---|---|---|---|---|---|
| v4 | 0.89 | 2.4% / 65% | 1.7% / 58% | 2.2% / 64% | 2.3% / 65% |
| v7 | 0.77 | 3.4% / 71% | 1.8% / 46% | 2.9% / 66% | 3.3% / 71% |
| v3 | 0.73 | 9.6% / 75% | 1.8% / 26% | 3.5% / 47% | 8.0% / 70% |

Mean over 1000 random calibration/test splits with 52 calibration questions each.

- The gate is a backstop, not a substitute for a grounded prompt: on v3 it reaches 2% wrong only by
  holding back three answers in four.
- The verifier is weaker with six passages (AUROC 0.77 on v7 vs 0.89 on v4), so for strict targets
  v4 + gate answers more questions than v7 + gate.
- The guarantee is in expectation. With only 52 calibration questions individual splits vary: the
  90th-percentile risk on v3 at α = 0.05 was 8.6%. Use more calibration data in production.
- Any model that maps (question, passages, answer) to a probability can replace the verifier.

## How the evaluation works

1. **Evaluation set.** 170 drafted questions over the handbook were filtered by independent model
   checks (evidence must be quoted verbatim; "unanswerable" questions must really be unanswerable)
   down to 116, then audited item by item. 12 unsound items were removed, with the reason for each
   kept in [`data/evalset_dropped.jsonl`](data/evalset_dropped.jsonl). Five types: fact, paraphrase,
   numeric, multi-hop comparison, and unanswerable.
2. **Metrics.** Evidence recall@k and MRR, answer correctness, faithfulness to the retrieved passages,
   correct refusals, false refusals, latency and token cost, each with a bootstrap confidence interval.
3. **Judge check.** Correctness is graded by an LLM judge (`openai/gpt-oss-120b`, a different model
   family from the generator `qwen/qwen3.8-27b`). Its verdicts were compared with blind grades from an
   independent LLM reviewer, also from a different model family than the generator and the judge:
   97.6% agreement, Cohen's κ = 0.93 (n = 42). This is a consistency check between models, not a human
   gold standard. `python -m rag_eval_kit.review` produces the sheets for a person to grade instead.

## Run it

```bash
pip install -e ".[dev]"
python data/download.py                              # fetch the pinned corpus (~1.7 MB)
export LLM_API_KEY=...                               # any OpenAI-compatible endpoint; OpenRouter by default
python -m rag_eval_kit.run configs/v0_baseline.yaml  # answers + verdicts + metrics in results/v0_baseline/
python -m rag_eval_kit.gate configs/v4_grounded_prompt.yaml
python -m rag_eval_kit.compare                       # results/COMPARISON.md
python -m rag_eval_kit.report                        # docs/index.html
```

Everything except the LLM calls runs on CPU (small ONNX embedding and reranking models).
Set `LLM_PROVIDER=groq`, or `LLM_BASE_URL` for vLLM or a local Ollama server, to swap providers.

### Reproduce the published numbers with no API key

Every model response behind the results is committed in `results/cache/llm/`, most of them with
the exact request that produced them. Replay them:

```bash
LLM_OFFLINE=1 python -m rag_eval_kit.run configs/v7_decompose.yaml
```

This makes no calls to any model and reproduces `answers.jsonl` byte for byte (about 15 minutes on a
laptop CPU, mostly reranking; the first run also downloads the small embedding and reranking models). A request that is not in the cache (a new
prompt, model or question) fails loudly instead of silently calling an API.

## Running it on your own assistant

The pipeline under test is two functions: retrieve passages for a question, and answer from them
(`src/rag_eval_kit/pipeline.py`, `retrieval.py`). To audit a different system, replace them with
calls to that system and keep the evaluation set format (question, reference answer, evidence
passages). Data from a private system should go only to models you are allowed to send it to;
the whole kit runs against a local model server.

## Limitations

- One corpus, one generator, one judge. The point of the kit is to measure *your* system; these
  numbers are an example, not a benchmark.
- The comparison subset has 14 questions, so its intervals are wide (v7: 43–82%).
- A few kept questions are weak: some have more than one valid answer, and yes/no "unanswerable"
  questions can fairly be answered "no, it isn't mentioned". They are listed in the audit notes in
  [`data/README.md`](data/README.md).
- Model hosts and free tiers change. Each run records the model and the host that served it;
  all runs were made in November 2026.

## License

Code: MIT. Evaluation data: CC BY-SA 4.0, derived from the
[GitLab Handbook](https://handbook.gitlab.com), © GitLab B.V. See [`data/README.md`](data/README.md).
