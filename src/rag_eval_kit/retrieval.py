"""Dense, BM25, hybrid (reciprocal rank fusion) and reranked retrieval. All CPU-only."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from .chunking import Chunk

CACHE = Path(".cache")
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-12-v2"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class DenseRetriever:
    def __init__(self, chunks: list[Chunk]):
        from fastembed import TextEmbedding
        self.chunks = chunks
        self.model = TextEmbedding(EMBED_MODEL, cache_dir=str(CACHE / "fastembed"))
        key = hashlib.sha256("\x00".join(c.chunk_id + c.text for c in chunks).encode()).hexdigest()[:16]
        f = CACHE / "embeddings" / f"{key}.npy"
        if f.exists():
            self.matrix = np.load(f)
        else:
            self.matrix = np.array(list(self.model.embed([c.text for c in chunks], batch_size=32)))
            self.matrix /= np.linalg.norm(self.matrix, axis=1, keepdims=True)
            f.parent.mkdir(parents=True, exist_ok=True)
            np.save(f, self.matrix)

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        q = next(iter(self.model.embed([QUERY_PREFIX + query])))
        q = q / np.linalg.norm(q)
        scores = self.matrix @ q
        top = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in top]


class BM25Retriever:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.bm25 = BM25Okapi([_tokens(c.text) for c in chunks])

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        scores = self.bm25.get_scores(_tokens(query))
        top = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in top]


class HybridRetriever:
    """Reciprocal rank fusion of dense and BM25 rankings."""

    def __init__(self, dense: DenseRetriever, bm25: BM25Retriever, k_rrf: int = 60, depth: int = 50):
        self.dense, self.bm25, self.k_rrf, self.depth = dense, bm25, k_rrf, depth
        self.chunks = dense.chunks

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        fused: dict[int, float] = {}
        for ranking in (self.dense.search(query, self.depth), self.bm25.search(query, self.depth)):
            for rank, (i, _) in enumerate(ranking):
                fused[i] = fused.get(i, 0.0) + 1.0 / (self.k_rrf + rank + 1)
        return sorted(fused.items(), key=lambda x: -x[1])[:k]


class RerankedRetriever:
    """Fetch `depth` candidates from a first-stage retriever, re-score with a cross-encoder."""

    def __init__(self, first_stage, depth: int = 30):
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        self.first, self.depth = first_stage, depth
        self.chunks = first_stage.chunks
        self.model = TextCrossEncoder(RERANK_MODEL, cache_dir=str(CACHE / "fastembed"))

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        cands = self.first.search(query, self.depth)
        scores = list(self.model.rerank(query, [self.chunks[i].text for i, _ in cands]))
        order = np.argsort(-np.array(scores))[:k]
        return [(cands[j][0], float(scores[j])) for j in order]


def build_retriever(chunks: list[Chunk], kind: str, rerank: bool = False):
    if kind == "dense":
        r = DenseRetriever(chunks)
    elif kind == "bm25":
        r = BM25Retriever(chunks)
    elif kind == "hybrid":
        r = HybridRetriever(DenseRetriever(chunks), BM25Retriever(chunks))
    else:
        raise ValueError(kind)
    return RerankedRetriever(r) if rerank else r
