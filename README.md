# RAG Eval Kit

Measure why a RAG assistant gives wrong answers, fix it one change at a time, and show the
number each change moved.

> Work in progress. Results, report and usage instructions will land here.

## What it does

1. **Builds an evaluation set** of real questions over a document collection, each with a
   reference answer and the passages that contain it, including questions the documents
   *cannot* answer.
2. **Scores a RAG pipeline** on retrieval (recall@k, MRR), answer correctness, faithfulness
   to the retrieved context, and correct refusals, with bootstrap confidence intervals.
3. **Checks the judge.** LLM-judge verdicts are compared against human labels before any
   score is trusted.
4. **Runs ablations.** Each fix (chunking, hybrid retrieval, reranking, prompting) is
   measured on its own, including the ones that don't help.

## Run it

```bash
pip install -e ".[dev]"
python data/download.py                              # fetch the pinned corpus (~1.7 MB)
export LLM_API_KEY=...                               # any OpenAI-compatible endpoint; Groq by default
python -m rag_eval_kit.run configs/v0_baseline.yaml  # answers + verdicts + metrics in results/v0_baseline/
python -m rag_eval_kit.compare                       # side-by-side table of every finished run
```

Everything except the two LLMs runs on CPU (small ONNX embedding and reranking models).
Model responses are cached on disk, so re-running a configuration is free and deterministic.
Point `LLM_BASE_URL` at OpenRouter, vLLM or a local Ollama server to swap providers.

## Demo corpus

A pinned slice of the public GitLab Handbook (benefits, time off, policies, expenses).
See [`data/README.md`](data/README.md) for source and license.

## License

Code: MIT. Evaluation data: CC BY-SA 4.0 (derived from the GitLab Handbook).
