# Data

## Corpus

A pinned slice of the [GitLab Handbook](https://handbook.gitlab.com): benefits, compensation,
time off, people policies, onboarding/offboarding, travel and expenses (94 pages, listed in
`corpus_manifest.txt`). It reads like a real company's internal knowledge base, which is
exactly what most RAG assistants are built on.

- Source: `gitlab.com/gitlab-com/content-sites/handbook` at commit `887ea219` (2026-11-17)
- License: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), © GitLab B.V.
- The pages are **not** stored here. Fetch them with `python data/download.py`.

## Evaluation set

`evalset.jsonl` contains 104 questions, their reference answers and the handbook passages
that hold each answer. Because it is derived from the handbook, it is shared under the same
**CC BY-SA 4.0** license, with attribution to GitLab B.V.

**v1.1 (104 items).** An audit of all 116 drafted items found 12 that were unsound. Four
had wrong reference answers, two were labelled unanswerable although the handbook answers
them, and six were not meaningful. They are kept, with the reason for each, in
`evalset_dropped.jsonl`, so the cut can be checked. Six of the twelve were multi-hop
items, so the multi-hop subset is now small (14) and its numbers carry wide intervals.

The code in this repository is MIT licensed (see `LICENSE`).
