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

**Kept, but weak.** These items are sound enough to keep, but a reader should know about them:

- `q018`, `q070`: more than one valid answer, only one reference.
- `q025`: "what percentage is contributed" can mean the team member's 5% or the 9% total.
- `q068`: the rule is Australia's, but the question does not name the country.
- `q078`, `q081`, `q090`: compare two things that do not line up, so they read unnaturally.
- `q088`: the Australian side of the reference is hard to state cleanly.
- `q022`, `q056`: typed as numeric but not really number questions.
- Yes/no unanswerable items such as `q093` and `q096` ("does the handbook say X?") can fairly be
  answered "no, it is not mentioned", which the judge may not count as a refusal.

These items are never used as showcase examples in the report.

The code in this repository is MIT licensed (see `LICENSE`).
