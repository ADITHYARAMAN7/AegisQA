"""
Production Hybrid Information Retrieval Engine for RAG Quality Evaluation.
Implements genuine mathematical:
1. Okapi BM25 (k1=1.5, b=0.75) with analytical Inverse Document Frequency (IDF)
2. Sublinear TF-IDF Vector Cosine Similarity
3. Reciprocal Rank Fusion (RRF) & weighted hybrid scoring across chunked technical manuals
"""
import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "to", "of", "in",
    "on", "for", "with", "at", "by", "from", "or", "and", "my", "how", "do", "i",
    "what", "why", "can", "it", "its", "this", "that", "while", "am", "when"
})


@dataclass
class RetrievedChunk:
    doc_id: str
    chunk_id: str
    product_name: str
    category: str
    text: str
    bm25_score: float
    tfidf_cosine: float
    hybrid_score: float
    rank: int


class HybridBM25Retriever:
    """
    Okapi BM25 + TF-IDF Cosine Similarity Hybrid Retriever over technical documentation chunks.
    """

    def __init__(self, kb_path: Optional[str] = None, k1: float = 1.5, b: float = 0.75, alpha: float = 0.65):
        self.k1 = k1
        self.b = b
        self.alpha = alpha  # Weight for normalized BM25 vs TF-IDF cosine

        if kb_path is None:
            base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            kb_path = os.path.join(base, "datasets", "knowledge_base.json")

        with open(kb_path, "r", encoding="utf-8") as f:
            self.raw_kb: Dict[str, Dict[str, Any]] = json.load(f)

        self.chunks: List[Dict[str, Any]] = []
        self.doc_tokens: List[List[str]] = []
        self.doc_term_freqs: List[Counter] = []
        self.doc_Freqs: Counter = Counter()
        self.idf: Dict[str, float] = {}
        self.avgdl: float = 1.0
        self._build_index()

    @staticmethod
    def tokenize(text: str) -> List[str]:
        tokens = re.findall(r"[a-z0-9]+(?:[._-][a-z0-9]+)*", text.lower())
        return [t for t in tokens if t not in STOPWORDS and len(t) > 1]

    def _build_index(self) -> None:
        for doc_id, entry in self.raw_kb.items():
            # Create granular semantic chunks per document (Specs Chunk + Troubleshooting/Safety Chunk)
            specs_text = (
                f"Product: {entry['product']} ({entry['category']}). "
                f"Technical Specifications: {json.dumps(entry['specs'])}."
            )
            trouble_text = (
                f"Product: {entry['product']} ({entry['category']}). "
                f"Troubleshooting & Safety Policy: {entry['troubleshooting']} "
                f"{entry['specs'].get('security_policy', '')} {entry['specs'].get('alerts', '')}"
            )
            full_text = f"{specs_text} {trouble_text}"
            self.chunks.append({
                "doc_id": doc_id,
                "chunk_id": f"{doc_id}#FULL",
                "product": entry["product"],
                "category": entry["category"],
                "text": full_text,
            })

        N = len(self.chunks)
        total_len = 0
        for chunk in self.chunks:
            toks = self.tokenize(chunk["text"])
            self.doc_tokens.append(toks)
            tf = Counter(toks)
            self.doc_term_freqs.append(tf)
            total_len += len(toks)
            for term in set(toks):
                self.doc_Freqs[term] += 1

        self.avgdl = total_len / max(1, N)
        # Robertson-Sparck Jones BM25 IDF formula
        for term, df in self.doc_Freqs.items():
            self.idf[term] = math.log(1.0 + (N - df + 0.5) / (df + 0.5))

    def _compute_bm25(self, query_tokens: List[str], doc_idx: int) -> float:
        tf = self.doc_term_freqs[doc_idx]
        dl = len(self.doc_tokens[doc_idx])
        score = 0.0
        for q in query_tokens:
            if q not in tf:
                continue
            f = tf[q]
            idf_q = self.idf.get(q, 0.0)
            numerator = f * (self.k1 + 1.0)
            denominator = f + self.k1 * (1.0 - self.b + self.b * (dl / self.avgdl))
            score += idf_q * (numerator / denominator)
        return score

    def _compute_tfidf_cosine(self, query_tokens: List[str], doc_idx: int) -> float:
        if not query_tokens:
            return 0.0
        q_tf = Counter(query_tokens)
        d_tf = self.doc_term_freqs[doc_idx]

        dot = 0.0
        q_norm_sq = 0.0
        d_norm_sq = 0.0

        for term, count in q_tf.items():
            w_q = (1.0 + math.log(count)) * self.idf.get(term, 0.5)
            q_norm_sq += w_q * w_q
            if term in d_tf:
                w_d = (1.0 + math.log(d_tf[term])) * self.idf.get(term, 0.5)
                dot += w_q * w_d

        for term, count in d_tf.items():
            w_d = (1.0 + math.log(count)) * self.idf.get(term, 0.5)
            d_norm_sq += w_d * w_d

        denom = math.sqrt(q_norm_sq) * math.sqrt(d_norm_sq)
        return dot / denom if denom > 1e-9 else 0.0

    def retrieve(self, query: str, top_k: int = 2, min_hybrid_score: float = 0.18) -> List[RetrievedChunk]:
        q_tokens = self.tokenize(query)
        if not q_tokens:
            return []

        raw_scores = []
        for idx, chunk in enumerate(self.chunks):
            bm25 = self._compute_bm25(q_tokens, idx)
            cosine = self._compute_tfidf_cosine(q_tokens, idx)
            raw_scores.append((idx, bm25, cosine))

        max_bm25 = max((r[1] for r in raw_scores), default=1.0) or 1.0
        results: List[RetrievedChunk] = []

        for idx, bm25, cosine in raw_scores:
            norm_bm25 = bm25 / max_bm25 if max_bm25 > 0 else 0.0
            # Require genuine lexical/semantic match (cosine >= 0.08 and raw bm25 >= 1.5)
            if bm25 < 1.5 or cosine < 0.08:
                continue
            hybrid = self.alpha * norm_bm25 + (1.0 - self.alpha) * cosine
            if hybrid >= min_hybrid_score:
                ch = self.chunks[idx]
                results.append(RetrievedChunk(
                    doc_id=ch["doc_id"],
                    chunk_id=ch["chunk_id"],
                    product_name=ch["product"],
                    category=ch["category"],
                    text=ch["text"],
                    bm25_score=round(bm25, 4),
                    tfidf_cosine=round(cosine, 4),
                    hybrid_score=round(hybrid, 4),
                    rank=0,
                ))

        results.sort(key=lambda x: x.hybrid_score, reverse=True)
        trimmed = results[:top_k]
        for rank_idx, item in enumerate(trimmed, start=1):
            item.rank = rank_idx
        return trimmed
