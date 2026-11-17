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

`evalset.jsonl` (added later) contains questions, reference answers and the IDs of the
passages that contain each answer. Because it is derived from the handbook, it is shared
under the same **CC BY-SA 4.0** license, with attribution to GitLab B.V.

The code in this repository is MIT licensed (see `LICENSE`).
