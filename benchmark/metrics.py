#!/usr/bin/env python3
# repo/benchmark/metrics.py
from __future__ import annotations

import math
import statistics
from typing import Any, Dict, Iterable, List, Tuple


# -----------------------------
# Text matching helpers
# -----------------------------

def _norm(s: str) -> str:
    return (s or "").strip().lower()

def contains_any(hay: str, needles: Iterable[str]) -> bool:
    h = _norm(hay)
    return any(_norm(n) in h for n in needles if n)

def jaccard(a: str, b: str) -> float:
    A = set(_norm(a).split())
    B = set(_norm(b).split())
    if not A and not B: return 1.0
    if not A or not B: return 0.0
    return len(A & B) / len(A | B)

def fuzzy_hit(text: str, gold_snippets: Iterable[str], thresh: float = 0.6) -> bool:
    t = _norm(text)
    for g in gold_snippets:
        if not g:
            continue
        g = _norm(g)
        if g in t:
            return True
        if jaccard(t, g) >= thresh:
            return True
    return False


# -----------------------------
# Metric computations (per-query)
# -----------------------------

def metric_extraction_quality(indexed_items: List[str], expected_answer: str) -> float:
    """
    Heuristic extraction quality:
    - For vector/keyword systems: did the expected answer text get indexed at all?
    - For graph memory: caller can pass in list of (subject|predicate|object) strings.
    Returns 1.0 if present (substring), else 0.0.
    """
    want = _norm(expected_answer)
    for it in indexed_items:
        if want and want in _norm(it):
            return 1.0
    return 0.0


def metric_retrieval_precision_k(retrieved_texts: List[str], relevant_texts: List[str], k: int) -> float:
    """
    Precision@k: fraction of top-k retrieved items that are relevant by fuzzy match.
    """
    if k <= 0:
        return 0.0
    top = retrieved_texts[:k]
    if not top:
        return 0.0
    hits = sum(1 for t in top if fuzzy_hit(t, relevant_texts))
    return hits / min(k, len(retrieved_texts))


def metric_retrieval_recall_k(retrieved_texts: List[str], relevant_texts: List[str], k: int) -> float:
    """
    Recall@k: fraction of relevant_texts that appear in top-k retrieved items (by fuzzy match).
    """
    if not relevant_texts:
        return 0.0
    top = retrieved_texts[:k]
    covered = 0
    for rel in relevant_texts:
        if any(fuzzy_hit(t, [rel]) for t in top):
            covered += 1
    return covered / len(relevant_texts)


def metric_update_accuracy(pred_answer: str, expected_answer: str, retrieved_texts: List[str], relevant_texts: List[str]) -> float:
    """
    Update accuracy:
      - 1 if the predicted answer equals expected_answer (post-update),
      - and (if there are conflicting relevant_texts) the correct snippet is present among retrieved
        and any clearly conflicting snippet is not ranked above the correct one.
      - else 0.
    Fallback: if we can't detect conflicts robustly, treat EM as proxy.
    """
    em = int(_norm(pred_answer) == _norm(expected_answer))
    # detect potential conflict: relevant_texts might include earlier + updated
    # We consider any relevant snippet containing the expected answer as "correct", others as "conflicts".
    correct_snips = [r for r in relevant_texts if _norm(expected_answer) in _norm(r)]
    conflicts = [r for r in relevant_texts if _norm(expected_answer) not in _norm(r)]
    if not conflicts or not retrieved_texts:
        return float(em)

    # If conflict exists, ensure at least one correct snippet is retrieved, and it appears before conflicts.
    top = [_norm(t) for t in retrieved_texts]
    first_correct = min((i for i, t in enumerate(top) if any(_norm(c) in t for c in correct_snips)), default=None)
    first_conflict = min((i for i, t in enumerate(top) if any(_norm(c) in t for c in conflicts)), default=None)

    if em and first_correct is not None and (first_conflict is None or first_correct <= first_conflict):
        return 1.0
    return float(em)


def metric_end2end_accuracy(pred_answer: str, expected_answer: str) -> float:
    """Exact match end-to-end response correctness."""
    return float(_norm(pred_answer) == _norm(expected_answer))


def metric_mrr(retrieved_texts: List[str], relevant_texts: List[str]) -> float:
    """
    Mean Reciprocal Rank for the first relevant hit.
    """
    for i, t in enumerate(retrieved_texts, start=1):
        if fuzzy_hit(t, relevant_texts):
            return 1.0 / i
    return 0.0


def metric_latency_ms(latencies: List[float]) -> Dict[str, float]:
    if not latencies:
        return {"avg_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
    return {
        "avg_ms": float(statistics.mean(latencies)),
        "p50_ms": float(statistics.median(latencies)),
        "p95_ms": float(_percentile(latencies, 95)),
        "max_ms": float(max(latencies)),
    }


def _percentile(xs: List[float], p: float) -> float:
    xs = sorted(xs)
    k = (len(xs) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return xs[int(k)]
    d0 = xs[f] * (c - k)
    d1 = xs[c] * (k - f)
    return d0 + d1


# -----------------------------
# Aggregation helpers
# -----------------------------

def safe_mean(xs: List[float]) -> float:
    return float(statistics.mean(xs)) if xs else 0.0

def aggregate_by_key(rows: List[Dict[str, Any]], key: str, metrics: List[str]) -> List[Dict[str, Any]]:
    """
    Group rows by `rows[i][key]` and average listed metric fields.
    """
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        k = r.get(key, "unknown")
        buckets.setdefault(k, []).append(r)
    out: List[Dict[str, Any]] = []
    for k, grp in buckets.items():
        agg: Dict[str, Any] = {key: k, "count": len(grp)}
        for m in metrics:
            agg[m] = safe_mean([float(g.get(m, 0.0)) for g in grp])
        out.append(agg)
    return sorted(out, key=lambda d: d[key])
