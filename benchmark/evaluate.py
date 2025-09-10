#!/usr/bin/env python3
# benchmark/evaluate.py
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

# Make sure project root is on sys.path if run as a script
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from loguru import logger
from ranx import Qrels, Run, evaluate as ranx_evaluate

# ---- memory systems ----
from memory.keyword_baseline import KeywordBaseline
from memory.vector_memory import VectorMemory
from memory.graph_memory import GraphMemory

try:
    from memory.third_party.mem0_adapter import Mem0Memory
except Exception:
    Mem0Memory = None
try:
    from memory.third_party.llamaindex_adapter import LlamaIndexMemory
except Exception:
    LlamaIndexMemory = None


# ---------------- small helpers ----------------

def norm(s: str) -> str:
    return (s or "").strip().lower()


def _stable_id(parts: List[str]) -> str:
    """Deterministic id for qrels/run doc ids across processes."""
    h = hashlib.sha1("::".join(parts).encode("utf-8")).hexdigest()
    return h


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    txt = path.read_text(encoding="utf-8").strip()
    if not txt:
        return []

    # JSON array
    try:
        obj = json.loads(txt)
        if isinstance(obj, list):
            return obj
    except Exception:
        pass

    # JSONL
    items: List[Dict[str, Any]] = []
    ok = True
    for line in txt.splitlines():
        ln = line.strip()
        if not ln:
            continue
        try:
            items.append(json.loads(ln))
        except Exception:
            ok = False
            break
    if ok and items:
        return items

    # Concatenated objects
    items = []
    buf = []
    depth = 0
    in_str = False
    esc = False
    for ch in txt:
        buf.append(ch)
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch in "{[":
                depth += 1
            elif ch in "}]":
                depth -= 1
                if depth == 0:
                    chunk = "".join(buf).strip()
                    items.append(json.loads(chunk))
                    buf.clear()
    if not items:
        raise ValueError(f"Could not parse dataset at {path}")
    return items


# ---------------- qrels/run & predictions ----------------

def build_qrels_and_run_for_query(
    conv_id: str,
    question_id: str,
    expected: str,
    relevant_texts: List[str],
    retrieved: List[Dict[str, Any]],
    k: int,
) -> Tuple[Dict[str, Dict[str, int]], Dict[str, Dict[str, float]]]:
    qid = f"{conv_id}::{question_id}"
    qrels: Dict[str, Dict[str, int]] = {qid: {}}
    run: Dict[str, Dict[str, float]] = {qid: {}}

    expected_l = norm(expected)
    rel_texts_l = [norm(t) for t in (relevant_texts or [])]

    def make_doc_id(it: Dict[str, Any], idx: int) -> str:
        if it.get("id"):
            return str(it["id"])
        parts = [
            str(it.get("type") or ""),
            str(it.get("subject") or ""),
            str(it.get("predicate") or ""),
            str(it.get("object") or ""),
            str(it.get("text") or ""),
        ]
        return f"{_stable_id(parts)}_{idx}"

    for idx, it in enumerate(retrieved[: max(k, 1)]):
        doc_id = make_doc_id(it, idx)
        score = float(it.get("score", 0.0))
        run[qid][doc_id] = score

        text_l = norm(it.get("text") or "")
        obj_l = norm(it.get("object") or "")
        is_rel = False
        if expected_l and (expected_l in text_l or expected_l == obj_l or expected_l in obj_l):
            is_rel = True
        if rel_texts_l:
            for rt in rel_texts_l:
                if rt and rt in text_l:
                    is_rel = True
                    break
        qrels[qid][doc_id] = 1 if is_rel else 0

    return qrels, run


def predict_answer_from_items(items: List[Dict[str, Any]]) -> str:
    if not items:
        return ""
    top = items[0]
    if (top.get("type") or "") == "fact":
        return norm(top.get("object", ""))
    return norm(top.get("text", ""))


# ---------------- system runner ----------------

@dataclass
class _PerQuery:
    qid: str
    scenario: str
    difficulty: str
    em: float
    hit_at_k: float
    rank: Optional[int]
    retrieved_count: int
    latency_ms: float
    pred: str


def run_system(
    system_name: str,
    dataset: List[Dict[str, Any]],
    *,
    top_k: int,
    index_backend: str,
    collection_or_table: str,
    vector_db_path: str,
    graph_db_path: str,
    mem0_chroma_path: str,
    llama_chroma_path: str,
    restrict_to_conv: bool,
    verbose: bool,
    print_k: int,
) -> Dict[str, Any]:

    # 1) Init memory
    if system_name == "keyword":
        mem = KeywordBaseline()
    elif system_name == "vector":
        mem = VectorMemory(
            name="vector_memory",
            index_backend=index_backend,
            collection_or_table=collection_or_table,
            persist_path=vector_db_path,
            restrict_to_conv=restrict_to_conv,
        )
    elif system_name == "graph":
        mem = GraphMemory(name="graph_memory", restrict_to_conv=restrict_to_conv)
    elif system_name == "mem0":
        if Mem0Memory is None:
            raise RuntimeError("Mem0 not available")
        mem = Mem0Memory(
            name="mem0_eval",
            collection=collection_or_table,
            restrict_to_conv=restrict_to_conv,
            chroma_path=mem0_chroma_path,
        )
    elif system_name == "llama":
        if LlamaIndexMemory is None:
            raise RuntimeError("Llama not available")
        mem = LlamaIndexMemory(
            name="llama_eval",
            collection=collection_or_table,
            restrict_to_conv=restrict_to_conv,
            chroma_path=llama_chroma_path,
        )
    else:
        raise ValueError(f"Unknown system '{system_name}'")

    logger.info(f"Running system={system_name}")

    all_qrels: Dict[str, Dict[str, int]] = {}
    all_run: Dict[str, Dict[str, float]] = {}
    per_query_rows: List[Dict[str, Any]] = []
    diag_rows: List[_PerQuery] = []

    for ci, c in enumerate(dataset):

        conv_id = c.get("conversation_id") or c.get("id") or f"conv_{ci}"
        scenario = c.get("scenario", "unknown")
        difficulty = c.get("difficulty", "unknown")

        for i, t in enumerate(c.get("turns", []), start=1):
            while True:
                try:
                    mem.add_turn(conv_id, i, t.get("role", ""), t.get("content", ""))

                    break
                except Exception as e:
                    print(e)
                    time.sleep(10)

        for qi, ev in enumerate(c.get("eval", []), start=1):
            q = (ev.get("question") or "").strip()
            expected = (ev.get("expected_answer") or "").strip()
            relevant_texts = ev.get("relevant_texts", []) or []

            if not q:
                continue

            start = time.perf_counter()
            out = mem.retrieve(conv_id, q, top_k=top_k)
            latency_ms = (time.perf_counter() - start) * 1000.0

            results = out.get("results", [])
            simple_items: List[Dict[str, Any]] = []
            for it in results:
                simple_items.append({
                    "type": it.get("type"),
                    "id": it.get("id"),
                    "subject": it.get("subject"),
                    "predicate": it.get("predicate"),
                    "object": it.get("object"),
                    "text": it.get("text") or it.get("object") or it.get("subject") or "",
                    "score": float(it.get("score", 0.0)),
                    "source": it.get("source"),
                })

            qrels_q, run_q = build_qrels_and_run_for_query(
                conv_id=conv_id,
                question_id=str(qi),
                expected=expected,
                relevant_texts=relevant_texts,
                retrieved=simple_items,
                k=top_k,
            )
            all_qrels.update(qrels_q)
            all_run.update(run_q)

            qid = f"{conv_id}::{qi}"
            pred = predict_answer_from_items(simple_items)
            em = 1.0 if norm(expected) == pred else 0.0

            # success log per query
            logger.success(
                f"[{system_name}] conv={conv_id} q{qi}: "
                f"Q='{q}' | gold='{expected}' | pred='{pred}' | EM={em} | latency={latency_ms:.1f}ms"
            )

            rank = None
            hit = 0.0
            qrels_this = qrels_q[qid]
            ranked_ids = sorted(run_q[qid].items(), key=lambda x: x[1], reverse=True)
            for r_idx, (doc_id, _score) in enumerate(ranked_ids, start=1):
                if qrels_this.get(doc_id, 0) > 0 and rank is None:
                    rank = r_idx
                    if r_idx <= top_k:
                        hit = 1.0
                    break

            diag_rows.append(_PerQuery(
                qid=qid, scenario=scenario, difficulty=difficulty, em=em,
                hit_at_k=hit, rank=rank, retrieved_count=len(simple_items),
                latency_ms=latency_ms, pred=pred
            ))

            per_query_rows.append({
                "qid": qid,
                "scenario": scenario,
                "difficulty": difficulty,
                "EM": em,
                "f1@K": 0.0, "hits@K": 0.0, "map": 0.0, "mrr": 0.0,
                "ndcg@K": 0.0, "precision@K": 0.0, "recall@K": 0.0,
            })

    # Ranx metrics
    qrels_all = Qrels(all_qrels)
    run_all = Run(all_run)
    metrics = {
        f"precision@{top_k}", f"recall@{top_k}", f"hits@{top_k}", f"f1@{top_k}",
        "map", "mrr", f"ndcg@{top_k}",
    }
    ranx_scores = ranx_evaluate(qrels_all, run_all, metrics=metrics)

    macro_em = sum(r["EM"] for r in per_query_rows) / max(len(per_query_rows), 1)

    macro = {
        "count": len(per_query_rows),
        "EM": macro_em,
        "precision@K": ranx_scores.get(f"precision@{top_k}", 0.0),
        "recall@K": ranx_scores.get(f"recall@{top_k}", 0.0),
        "hits@K": ranx_scores.get(f"hits@{top_k}", 0.0),
        "f1@K": ranx_scores.get(f"f1@{top_k}", 0.0),
        "map": ranx_scores.get("map", 0.0),
        "mrr": ranx_scores.get("mrr", 0.0),
        "ndcg@K": ranx_scores.get(f"ndcg@{top_k}", 0.0),
        "avg_latency_ms": sum(d.latency_ms for d in diag_rows) / max(len(diag_rows), 1),
        "avg_retrieved": sum(d.retrieved_count for d in diag_rows) / max(len(diag_rows), 1),
    }

    return {
        "macro": macro,
        "per_query": per_query_rows,
        "diagnostics": [d.__dict__ for d in diag_rows],
    }


# ---------------- CLI ----------------

def parse_args():
    ap = argparse.ArgumentParser(description="Evaluate memory systems on conversation datasets.")
    ap.add_argument("--data", default="datasets/conversations.jsonl")
    ap.add_argument("--compare", default="keyword,vector,graph")
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--index_backend", default="chroma", choices=["chroma", "lancedb"])
    ap.add_argument("--collection_or_table", default="memorybridge_facts")
    ap.add_argument("--vector_db_path", default=".memdb/vector_chroma")
    ap.add_argument("--graph_db_path", default=".memdb/graph_chroma")
    ap.add_argument("--mem0_chroma_path", default=".memdb/mem0_chroma_eval")
    ap.add_argument("--llama_chroma_path", default=".memdb/llamaindex_chroma_eval")
    ap.add_argument("--restrict_to_conv", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--print_k", type=int, default=5)
    ap.add_argument("--results_file", default="results.json")
    return ap.parse_args()


def main():
    args = parse_args()
    dataset = load_dataset(Path(args.data))
    systems = [s.strip().lower() for s in args.compare.split(",") if s.strip()]

    results_by_system = {}
    for sys_name in systems:
        results_by_system[sys_name] = run_system(
            sys_name,
            dataset,
            top_k=args.top_k,
            index_backend=args.index_backend,
            collection_or_table=args.collection_or_table,
            vector_db_path=args.vector_db_path,
            graph_db_path=args.graph_db_path,
            mem0_chroma_path=args.mem0_chroma_path,
            llama_chroma_path=args.llama_chroma_path,
            restrict_to_conv=args.restrict_to_conv,
            verbose=args.verbose,
            print_k=args.print_k,
        )

    for sys_name, res in results_by_system.items():
        macro = res["macro"]
        logger.info(f"\nSystem={sys_name} Results:")
        logger.info(json.dumps(macro, indent=2))

    if args.results_file:
        with open(args.results_file, "w", encoding="utf-8") as f:
            json.dump(results_by_system, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved results to {args.results_file}")


if __name__ == "__main__":
    main()
