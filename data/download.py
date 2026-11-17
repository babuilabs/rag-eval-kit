"""Download the evaluation corpus: a pinned slice of the public GitLab Handbook.

The handbook text is licensed CC BY-SA 4.0 by GitLab B.V. It is not stored in this
repository; this script fetches the exact pages listed in corpus_manifest.txt at a fixed
commit, so every run sees the same documents.

    python data/download.py
"""
from __future__ import annotations

import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

PROJECT = "gitlab-com/content-sites/handbook"
COMMIT = "887ea219e1c082aa2fed20ece0a7129a369350d0"  # main @ 2026-11-17
PREFIX = "content/handbook/"

HERE = Path(__file__).parent
OUT = HERE / "corpus"


def raw_url(rel: str) -> str:
    path = urllib.parse.quote(PREFIX + rel, safe="/")
    return f"https://gitlab.com/{PROJECT}/-/raw/{COMMIT}/{path}"


def main() -> int:
    pages = [l.strip() for l in (HERE / "corpus_manifest.txt").read_text().splitlines() if l.strip()]
    OUT.mkdir(exist_ok=True)
    failed = []
    for i, rel in enumerate(pages, 1):
        dest = OUT / rel
        if dest.exists() and dest.stat().st_size > 0:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(raw_url(rel), headers={"User-Agent": "rag-eval-kit"})
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=30) as r:
                    dest.write_bytes(r.read())
                break
            except Exception as e:  # network hiccup: back off and retry
                if attempt == 2:
                    failed.append(f"{rel}: {e}")
                time.sleep(2 * (attempt + 1))
        print(f"[{i}/{len(pages)}] {rel}")
    total = sum(p.stat().st_size for p in OUT.rglob("*.md"))
    print(f"{len(pages) - len(failed)} pages, {total / 1e6:.1f} MB in {OUT}")
    for f in failed:
        print("FAILED", f, file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
