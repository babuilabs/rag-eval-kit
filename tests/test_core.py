from rag_eval_kit.chunking import fixed_chunks, heading_chunks
from rag_eval_kit.corpus import Document, clean_markdown, evidence_overlap, normalize
from rag_eval_kit.metrics import bootstrap_ci, question_score, retrieval_scores

DOC = Document("x/page.md", "Leave", """Intro paragraph that is long enough to stand on its own as a section of text here.

## Parental leave

Team members in the UK get 16 weeks of paid parental leave. """ + "More detail. " * 30 + """

## Sick leave

Short.

### Notes

Sick leave in Australia is 10 days per year. """ + "Extra words. " * 30)


def test_clean_markdown_strips_front_matter_and_shortcodes():
    title, text = clean_markdown('---\ntitle: "Benefits"\n---\nHello {{< note >}} world <!-- hidden -->')
    assert title == "Benefits"
    assert "note" not in text and "hidden" not in text and "Hello" in text


def test_normalize_drops_markdown_and_links():
    assert normalize("**Paid** [leave](https://x.y) is `16` weeks") == "paid leave is 16 weeks"


def test_evidence_overlap_full_and_partial():
    passage = "Team members in the UK get 16 weeks of paid parental leave and more."
    assert evidence_overlap("Team members in the UK get 16 weeks of paid parental leave", passage) == 1.0
    assert evidence_overlap("Team members in France get 20 weeks of something else entirely", passage) < 0.5


def test_fixed_chunks_cover_text():
    chunks = fixed_chunks(DOC, size=200)
    assert all(len(c.text) <= 200 for c in chunks)
    assert sum(len(c.text) for c in chunks) >= len(DOC.text) - len(chunks) * 2


def test_heading_chunks_keep_heading_path_and_merge_tiny_sections():
    chunks = heading_chunks(DOC, max_chars=1500, min_chars=50)
    assert any(c.text.startswith("[Leave > Parental leave]") for c in chunks)
    # "Short." is merged forward into the Notes section rather than standing alone
    assert not any(c.text.strip().endswith("Short.") for c in chunks)
    assert any("Sick leave in Australia" in c.text for c in chunks)


def test_retrieval_scores_recall_and_rank():
    item = {"evidence": [{"doc_id": "a", "quote": "Team members in the UK get 16 weeks of paid parental leave"},
                         {"doc_id": "b", "quote": "Sick leave in Australia is 10 days per year"}]}
    retrieved = ["unrelated text about travel budgets and hotels",
                 "Team members in the UK get 16 weeks of paid parental leave."]
    s = retrieval_scores(item, retrieved)
    assert s["recall"] == 0.5 and s["rr"] == 0.5 and not s["all_found"]


def test_question_score():
    assert question_score({"type": "fact"}, {"correctness": "partial"}) == 0.5
    assert question_score({"type": "unanswerable"}, {"refused": True}) == 1.0
    assert question_score({"type": "unanswerable"}, {"refused": False}) == 0.0


def test_bootstrap_ci_brackets_mean():
    lo, hi = bootstrap_ci([0, 1] * 50)
    assert lo < 0.5 < hi


def _rows(pairs):
    return [{"p_supported": p, "loss": l, "shown": True, "type": "fact", "verdict": {}} for p, l in pairs]


def test_gate_risk_falls_as_threshold_rises():
    from rag_eval_kit.gate import coverage, risk
    rows = _rows([(0.9, 0.0), (0.8, 0.0), (0.3, 1.0), (0.2, 1.0)])
    assert risk(rows, 0.0) == 0.5 and coverage(rows, 0.0) == 1.0
    assert risk(rows, 0.5) == 0.0 and coverage(rows, 0.5) == 0.5


def test_crc_threshold_meets_adjusted_risk_bound():
    from rag_eval_kit.gate import crc_threshold, risk
    rows = _rows([(0.95, 0.0)] * 40 + [(0.4, 1.0)] * 10)
    tau = crc_threshold(rows, 0.1)
    assert (len(rows) * risk(rows, tau) + 1) / (len(rows) + 1) <= 0.1
    assert tau <= 0.95  # still answers the well-supported ones


def test_crc_threshold_answers_nothing_when_alpha_is_unreachable():
    from rag_eval_kit.gate import coverage, crc_threshold
    rows = _rows([(0.9, 0.0)] * 5)
    assert coverage(rows, crc_threshold(rows, 0.1)) == 0.0  # 1/(n+1) > alpha with n=5
