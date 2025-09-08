# benchmark/evaluate.py
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
from memory.third_party.mem0_adapter import Mem0Memory
from memory.third_party.llamaindex_adapter import LlamaIndexMemory


# ---- small helpers ----

def norm(s: str) -> str:
    return (s or "").strip().lower()


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    """
    Robust loader that supports:
      1) JSONL (one object per line)
      2) A single JSON array of objects
      3) Concatenated pretty-printed JSON objects (brace-balanced stream)
    """
    txt = path.read_text(encoding="utf-8").strip()
    if not txt:
        return []

    # Try as a whole JSON value first (array)
    try:
        obj = json.loads(txt)
        if isinstance(obj, list):
            return obj
    except Exception:
        pass

    # Try JSONL (one object per non-empty line)
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

    # Fallback: concatenated objects (brace counting)
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
        raise ValueError(f"Could not parse dataset at {path}. Make sure it is JSONL, a JSON array, or concatenated JSON objects.")
    return items


def build_qrels_and_run_for_query(
        conv_id: str,
        question_id: str,
        expected: str,
        relevant_texts: List[str],
        retrieved: List[Dict[str, Any]],
        k: int,
) -> Tuple[Dict[str, Dict[str, int]], Dict[str, Dict[str, float]]]:
    """
    Convert one query's retrieved items into ranx-compatible qrels & run.

    qrels: {qid: {doc_id: relevance}}
    run:   {qid: {doc_id: score}}

    Heuristic:
      - doc_id is formed from item.id if present, otherwise a stable hash of (type|subject|predicate|object|text)
      - relevant if:
          a) expected string appears in item.text OR item.object, or
          b) any relevant_texts string appears in item.text
    """
    qid = f"{conv_id}::{question_id}"
    qrels: Dict[str, Dict[str, int]] = {qid: {}}
    run: Dict[str, Dict[str, float]] = {qid: {}}

    expected_l = norm(expected)
    rel_texts_l = [norm(t) for t in (relevant_texts or [])]

    def make_doc_id(it: Dict[str, Any], idx: int) -> str:
        if it.get("id"):
            return str(it["id"])
        # Stable synthetic id
        parts = [
            it.get("type") or "",
            it.get("subject") or "",
            it.get("predicate") or "",
            it.get("object") or "",
            it.get("text") or "",
        ]
        j = "::".join(parts)
        return f"{hash(j)}_{idx}"

    for idx, it in enumerate(retrieved[: max(k, 1)]):
        doc_id = make_doc_id(it, idx)
        score = float(it.get("score", 0.0))
        run[qid][doc_id] = score

        # relevance check
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
    """
    Very simple "answer picker":
      - If top item is a fact, return its 'object'
      - Else return its 'text'
    """
    if not items:
        return ""
    top = items[0]
    if top.get("type") == "fact":
        return norm(top.get("object", ""))
    return norm(top.get("text", ""))


def _fmt_num(x, nd=3):
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def _agg_rows(rows: List[Dict[str, Any]], metrics_cols: List[str], count_col="count"):
    if not rows:
        return {}
    out = {count_col: len(rows)}
    for m in metrics_cols:
        vals = [float(r.get(m, 0.0)) for r in rows]
        out[m] = sum(vals) / max(len(vals), 1)
    return out


def _group_agg(rows: List[Dict[str, Any]], key: str, metrics_cols: List[str]):
    groups = defaultdict(list)
    for r in rows:
        groups[r.get(key, "unknown")].append(r)
    table = []
    for g in sorted(groups.keys()):
        agg = _agg_rows(groups[g], metrics_cols)
        agg[key] = g
        table.append(agg)
    return table


def _print_table(title, rows, metrics_cols, extra_cols=None, sys_name=None):
    if sys_name:
        print(f"\n== {title} (system={sys_name}) ==")
    else:
        print(f"\n== {title} ==")
    if not rows:
        print("(no data)")
        return
    cols = (extra_cols or []) + ["count"] + metrics_cols
    header = "  ".join(f"{c:>12}" for c in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        line_vals = []
        for c in cols:
            v = r.get(c, "")
            if c in {"scenario", "difficulty"}:
                line_vals.append(f"{str(v):>12}")
            elif c == "count":
                try:
                    line_vals.append(f"{int(v):>12d}")
                except Exception:
                    line_vals.append(f"{str(v):>12}")
            else:
                line_vals.append(f"{_fmt_num(v):>12}")
        print("  ".join(line_vals))


def print_per_system_reports(results_by_system: Dict[str, Dict[str, Any]]):
    metric_cols = ["EM", "f1@K", "hits@K", "map", "mrr", "ndcg@K", "precision@K", "recall@K"]
    for sys_name, pack in results_by_system.items():
        macro = pack.get("macro", {})
        perq = pack.get("per_query", [])

        # Overall
        overall_row = [{
            "count": macro.get("count", len(perq)),
            **{m: macro.get(m, 0.0) for m in metric_cols}
        }]
        _print_table("Overall", overall_row, metrics_cols=metric_cols, sys_name=sys_name)

        # By scenario
        by_scen = _group_agg(perq, key="scenario", metrics_cols=metric_cols)
        _print_table("By scenario", by_scen, metrics_cols=metric_cols, extra_cols=["scenario"], sys_name=sys_name)

        # By difficulty
        by_diff = _group_agg(perq, key="difficulty", metrics_cols=metric_cols)
        _print_table("By difficulty", by_diff, metrics_cols=metric_cols, extra_cols=["difficulty"], sys_name=sys_name)


# ---- system runner ----

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
    """
    Execute one memory system over the dataset and produce:
      - macro ranx metrics + EM
      - per-query rows for group aggregation
    """
    # 1) Construct memory
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
        mem = GraphMemory(
            name="graph_memory",
            index_backend=index_backend,
            collection_or_table=collection_or_table,
            persist_path=graph_db_path,
            graph_db_path=os.path.join(os.path.dirname(graph_db_path), "graph_evaluate.sqlite"),
            restrict_to_conv=restrict_to_conv,
        )
    elif system_name == "mem0":
        # Mem0 adapter; reads models/base_url/api_key from src.config.load_config()
        mem = Mem0Memory(
            name="mem0_memory_eval",
            collection=collection_or_table,
            restrict_to_conv=restrict_to_conv,
            chroma_path=mem0_chroma_path,
        )
    elif system_name == "llama":
        # LlamaIndex adapter; reads models/base_url/api_key from src.config.load_config()
        mem = LlamaIndexMemory(
            name="llamaindex_memory_eval",
            collection=collection_or_table,
            restrict_to_conv=restrict_to_conv,
            chroma_path=llama_chroma_path,
        )
    else:
        raise ValueError(f"Unknown system '{system_name}'")

    logger.info(f"Running system={system_name}")

    # 2) Build global qrels & run accumulators
    all_qrels: Dict[str, Dict[str, int]] = {}
    all_run: Dict[str, Dict[str, float]] = {}
    per_query_rows: List[Dict[str, Any]] = []

    for c in dataset:
        conv_id = c.get("conversation_id") or c.get("id") or "unknown_conv"
        scenario = c.get("scenario", "unknown")
        difficulty = c.get("difficulty", "unknown")

        # Ingest turns
        for i, t in enumerate(c.get("turns", []), start=1):
            mem.add_turn(conv_id, i, t.get("role", ""), t.get("content", ""))

        # Evaluate questions for this conversation
        for qi, ev in enumerate(c.get("eval", []), start=1):
            q = ev.get("question", "")
            expected = ev.get("expected_answer", "")
            relevant_texts = ev.get("relevant_texts", []) or []

            start = time.perf_counter()
            out = mem.retrieve(conv_id, q, top_k=top_k)
            latency_ms = (time.perf_counter() - start) * 1000.0
            _ = latency_ms  # reserved for future latency reporting

            # Normalize retrieved items into a simpler shape for use here
            results = out.get("results", [])
            simple_items = []
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

            # Verbose: per-query printout
            if verbose:
                print(f"\n[{system_name}] conv={conv_id}")
                print(f"Q: {q}")
                pred = predict_answer_from_items(simple_items)
                print(f"gold: {expected!r}   pred: {pred!r}   EM={1 if norm(expected) == pred else 0}")
                for rank, it in enumerate(simple_items[:max(print_k, 1)], start=1):
                    label = it["type"] or "item"
                    shown = it[
                        "text"] if label != "fact" else f"{it.get('subject')} | {it.get('predicate')} | {it.get('object')}"
                    print(f"  #{rank:02d}  {label:<10} score={it['score']:.4f}  {shown}")

            # qrels & run for this one query
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

            # per-query row (for grouping/averaging); EM + placeholders (ranx macro done later)
            per_query_rows.append({
                "scenario": scenario,
                "difficulty": difficulty,
                "EM": 1.0 if norm(expected) == predict_answer_from_items(simple_items) else 0.0,
                # The following are filled later as macro (groupwise via re-evaluate)
                "f1@K": 0.0, "hits@K": 0.0, "map": 0.0, "mrr": 0.0,
                "ndcg@K": 0.0, "precision@K": 0.0, "recall@K": 0.0,
            })

    # 3) Compute ranx metrics for ALL queries (macro)
    qrels_all = Qrels(all_qrels)
    run_all = Run(all_run)

    # Note: the k used for @K metrics is the same as top_k in args
    metrics = {
        f"precision@{top_k}",
        f"recall@{top_k}",
        f"hits@{top_k}",
        f"f1@{top_k}",
        "map",
        "mrr",
        f"ndcg@{top_k}",
    }
    ranx_scores = ranx_evaluate(qrels_all, run_all, metrics=metrics)

    # 4) Compute macro EM
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
    }

    # 5) Groupwise macro prints (scenario, difficulty)
    def group_macro(group_key: str) -> List[Dict[str, Any]]:
        rows = []
        groups = defaultdict(list)
        created_qids = list(qrels_all.qrels.keys())

        for idx, pq in enumerate(per_query_rows):
            g = pq.get(group_key, "unknown")
            groups[g].append(created_qids[idx])

        for g, qids in groups.items():
            # slice qrels/run by qids
            sub_qrels = {qid: qrels_all.qrels[qid] for qid in qids}
            sub_run = {qid: run_all.run[qid] for qid in qids if qid in run_all.run}
            if not sub_run:
                rows.append({
                    group_key: g, "count": len(qids),
                    "EM": sum(per_query_rows[i]["EM"] for i, q in enumerate(created_qids) if q in qids) / max(len(qids),
                                                                                                              1),
                    "precision@K": 0.0, "recall@K": 0.0, "hits@K": 0.0, "f1@K": 0.0,
                    "map": 0.0, "mrr": 0.0, "ndcg@K": 0.0
                })
                continue
            scores = ranx_evaluate(Qrels(sub_qrels), Run(sub_run), metrics=metrics)
            em = sum(per_query_rows[i]["EM"] for i, q in enumerate(created_qids) if q in qids) / max(len(qids), 1)
            rows.append({
                group_key: g, "count": len(qids),
                "EM": em,
                "precision@K": scores.get(f"precision@{top_k}", 0.0),
                "recall@K": scores.get(f"recall@{top_k}", 0.0),
                "hits@K": scores.get(f"hits@{top_k}", 0.0),
                "f1@K": scores.get(f"f1@{top_k}", 0.0),
                "map": scores.get("map", 0.0),
                "mrr": scores.get("mrr", 0.0),
                "ndcg@K": scores.get(f"ndcg@{top_k}", 0.0),
            })

        # pretty print like your earlier style:
        print(f"\n== Overall by system ({system_name}) ==")
        print(
            f"count={macro['count']}  EM={_fmt_num(macro['EM'])}  f1@{top_k}={_fmt_num(macro['f1@K'])}  hits@{top_k}={_fmt_num(macro['hits@K'])}  map={_fmt_num(macro['map'])}  mrr={_fmt_num(macro['mrr'])}  ndcg@{top_k}={_fmt_num(macro['ndcg@K'])}  precision@{top_k}={_fmt_num(macro['precision@K'])}  recall@{top_k}={_fmt_num(macro['recall@K'])}")
        print(f"\n== By system × {group_key} ({system_name}) ==")
        for r in rows:
            print(
                f"{group_key}={r[group_key]:<12} count={r['count']:>3}  EM={_fmt_num(r['EM'])}  f1@{top_k}={_fmt_num(r['f1@K'])}  hits@{top_k}={_fmt_num(r['hits@K'])}  map={_fmt_num(r['map'])}  mrr={_fmt_num(r['mrr'])}  ndcg@{top_k}={_fmt_num(r['ndcg@K'])}  precision@{top_k}={_fmt_num(r['precision@K'])}  recall@{top_k}={_fmt_num(r['recall@K'])}")

        return rows

    # Produce compact prints like your earlier style:
    _ = group_macro("scenario")
    _ = group_macro("difficulty")

    return {
        "macro": macro,
        "per_query": per_query_rows,
    }


# ---- CLI ----

def parse_args():
    ap = argparse.ArgumentParser(description="Evaluate memory systems on conversation datasets.")
    ap.add_argument("--data",default="datasets/conversations.jsonl", help="Path to JSON/JSONL dataset.")
    ap.add_argument(
        "--compare",
        default="keyword,vector,graph",
        help="Comma-separated systems: keyword,vector,graph,mem0,llama",
    )
    ap.add_argument("--top_k", type=int, default=3, help="Top-K to retrieve/evaluate.")
    ap.add_argument("--index_backend", default="chroma", choices=["chroma", "lancedb"],
                    help="Vector backend for vector/graph memories.")
    ap.add_argument("--collection_or_table", default="memorybridge_facts",
                    help="Collection/table name for vector stores.")
    ap.add_argument("--vector_db_path", default=".memdb/vector_chroma", help="Persist dir for VectorMemory.")
    ap.add_argument("--graph_db_path", default=".memdb/graph_chroma",
                    help="Persist dir for GraphMemory's internal vector index.")
    ap.add_argument("--mem0_chroma_path", default=".memdb/mem0_chroma_eval",
                    help="Persist dir for Mem0Memory's Chroma store.")
    ap.add_argument("--llama_chroma_path", default=".memdb/llamaindex_chroma_eval",
                    help="Persist dir for LlamaIndexMemory's Chroma store.")
    ap.add_argument("--restrict_to_conv", action="store_true", help="Scope retrieval to the same conversation.")
    ap.add_argument("--verbose", action="store_true", help="Print per-query details.")
    ap.add_argument("--print_k", type=int, default=5,
                    help="How many retrieved items to print per query when --verbose.")
    ap.add_argument("--results_file", default=None, help="Optional path to save results as JSON.")
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

    # Final clean, per-system reports (overall + scenario + difficulty)
    print_per_system_reports(results_by_system)
    # Save results to file if requested

    if args.results_file:
        import json
    with open(args.results_file, "w", encoding="utf-8") as f:
        json.dump(results_by_system, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved results to {args.results_file}")


if __name__ == "__main__":
    main()
