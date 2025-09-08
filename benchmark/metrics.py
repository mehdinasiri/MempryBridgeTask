# benchmark/metrics.py
from __future__ import annotations

from typing import Dict, List, Tuple, Any
from ranx import Qrels, Run, evaluate


def _to_ranx(
    qrels_dict: Dict[str, Dict[str, float]],
    run_dict: Dict[str, Dict[str, float]],
) -> Tuple[Qrels, Run]:
    """
    Convert plain dicts to ranx objects.

    qrels_dict format: {qid: {doc_id: gain, ...}, ...}
    run_dict   format: {qid: {doc_id: score, ...}, ...}
    """
    return Qrels(qrels_dict), Run(run_dict)


def evaluate_with_ranx(
    qrels_dict: Dict[str, Dict[str, float]],
    run_dict: Dict[str, Dict[str, float]],
    metrics: List[str],
) -> Tuple[Dict[str, float], Dict[str, Any], Dict[str, Dict[str, float]]]:
    """
    Compute macro metrics with ranx and also return per-query metrics
    (looping over qids since your ranx version does not expose per_query).

    Returns:
      macro_summary: overall metrics across all queries
      em_cov: {} placeholder to keep legacy return signature
      per_query: {qid: {metric: value}}
    """
    qrels, run = _to_ranx(qrels_dict, run_dict)

    # Macro (overall) metrics in one shot
    macro_summary = evaluate(qrels, run, metrics)

    # Per-query metrics: eval each qid independently
    per_query: Dict[str, Dict[str, float]] = {}
    for qid, rels in qrels_dict.items():
        sub_qrels = Qrels({qid: rels})
        sub_run = Run({qid: run_dict.get(qid, {})})
        per_query[qid] = evaluate(sub_qrels, sub_run, metrics)

    em_cov: Dict[str, Any] = {}
    return macro_summary, em_cov, per_query
