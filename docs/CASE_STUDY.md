# Case study: auditing a handbook chatbot

## The situation

A company puts its employee handbook behind a chat assistant. It is built the way most tutorials
show: split the documents every 1000 characters, embed them, fetch the three closest chunks, and ask
an LLM to answer from them. In demos it looks fine. In use, people start getting confident answers
that are wrong, such as the wrong number of weeks of leave or another country's pension rule, and
nobody can say how often it happens or which fix would help.

This repository recreates that situation on a public stand-in, 94 pages of the GitLab Handbook
(benefits, leave, policies, expenses), and runs the audit we would run on a client's system.

## Step 1: decide what "wrong" means, then measure it

Before changing anything, we built an evaluation set of 104 questions with reference answers and
the exact passages that contain them. It covers five kinds of questions: direct facts, paraphrased
questions, numbers and dates, comparisons across two countries, and 23 questions the handbook
cannot answer, where the only right response is "I don't know".

The set was audited item by item before it was trusted. 12 of 116 drafted items were removed: four
had wrong reference answers, two were labelled unanswerable although the handbook answers them, and
six were not meaningful. The reason for each removal is published. Without this step, part of the
"failures" on comparison questions would have been the test's fault, not the assistant's.

Answers are graded by an LLM judge from a different model family than the assistant. Its verdicts
were checked against blind grades from an independent reviewer model, also from a different family
(κ = 0.93). An early version of the judge contradicted itself on refusals; the check caught it and
the judge was fixed before any result below was produced.

**Baseline:** 67% of answerable questions answered correctly, and **a wrong answer shown on 20.7% of
all questions**: roughly one question in five got a confident wrong reply.

## Step 2: change one thing at a time

| Change | What moved |
|---|---|
| Chunks that follow headings | Evidence recall 61% → 86%. The biggest single gain. |
| Hybrid keyword + vector search | Nothing useful on this corpus, slightly worse. Not recommended here. |
| Cross-encoder reranker | Faithfulness 91–93% → 97%. |
| Grounded prompt (cite, or say "I don't know") | Wrong answers shown 9.6% → 2.4%. Never answered an unanswerable question. But it refused 16% of answerable ones. |
| Six passages instead of three | False refusals 16% → 10%, comparison questions 25% → 46%. |
| One search per compared item | Comparison questions 46% → 64%. |

Each row is a separate run on the same questions, with confidence intervals. The over-cautious
refusals after the prompt change were investigated, not tuned away: in 12 of 13 cases the right
passage had never been retrieved, so refusing was the correct behaviour and the fix belonged in
retrieval. That is what the last two rows did.

## Step 3: a safety gate, and an honest trade-off

For teams that need fewer wrong answers than any prompt delivers, we added a gate. A third model
checks whether each claim in an answer is supported by the passages, and a threshold set with
conformal risk control holds back answers that are likely wrong, to be routed to a person. On the
grounded-prompt version, a 3% target gave **1.7% wrong answers shown while still answering 58% of
questions directly**.

The gate is not free, and the report says so:

- It works well only on top of a grounded prompt. Applied to the earlier version, it reached the same
  safety only by holding back three answers in four.
- With more passages, the checker became less reliable, so the most accurate version is not the best
  base for the strictest target.
- With 104 questions the guarantee holds on average, not on every sample. A production deployment
  should calibrate it on more of its own traffic.

## The outcome

Three operating points, all measured, for the client to choose from:

| Choose | Right answers delivered | Wrong answers shown |
|---|---|---|
| Most answers | 68% | 3.4% |
| Balanced | 63% | 2.4% |
| Fewest wrong answers | 56% (rest to a person) | 1.7% |

against a starting point of 52% right and 20.7% wrong. Which one is right depends on what a wrong
answer costs the business, and that decision belongs to the client, not to us.

## What a client gets from the same audit

- An evaluation set built from their own documents, audited, with the audit trail.
- The same table for their system: what is failing (retrieval or answering), which fixes help, which
  do not, with confidence intervals.
- Fixes applied one at a time, each with its measured effect.
- Optionally, a gate with a stated target for wrong answers and the share routed to people.
- Everything reproducible from cached model responses, and able to run entirely on their own
  infrastructure.

## Limits of this demonstration

One corpus, one assistant model, one judge model. The comparison subset has only 14 questions.
Grading is by models, cross-checked by a model from a different family, not by people. These are
the right limits for a public demonstration; a client audit uses the client's documents, real
questions and, where it matters, their own reviewers.
