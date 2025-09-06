#!/usr/bin/env python3
# repo/benchmark/evaluate.py
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from utils.factory import (
    make_keyword_baseline,
    make_vector_memory,
    make_graph_memory,
    make_hooks,
)

from benchmark.metrics import (
    metric_extraction_quality,
    metric_retrieval_precision_k,
    metric_retrieval_recall_k,
    metric_update_accuracy,
    metric_mrr,
    metric_latency_ms,
    aggregate_by_key,
)

# Reuse naive answer extraction rules from demos
import re
WORK_AT_RE = re.compile(r"\bwork(?:s)?\s+at\s+([A-Za-z][A-Za-z0-9 .&-]+)", re.I)
LIVE_IN_RE = re.compile(r"\blive(?:s)?\s+in\s+([A-Za-z][A-Za-z0-9 .&-]+)", re.I)
NAME_IS_RE = re.compile(r"\b(?:my\s+name\s+is|people\s+call\s+me)\s+([A-Za-z][A-Za-z0-9 .&-]+)", re.I)


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


def extract_answer(query: str, retrieved_items: List[Dict[str, Any]]) -> str:
    """
    Preference: fact → edge → regex-from-chunk
    """
    ql = (query or "").lower()

    # Facts (graph memory)
    for it in retrieved_items:
        if it.get("type") == "fact":
            pred = (it.get("predicate") or "").lower()
            obj = (it.get("object") or "").lower()
            if "work" in ql and pred in ("works_at", "work_at", "employer"):
                return obj
            if ("city" in ql or "live" in ql) and pred in ("lives_in", "live_in", "location"):
                return obj
            if "name" in ql and pred in ("name_is", "name"):
                return obj

    # Graph edges
    for it in retrieved_items:
        if it.get("type") == "graph_edge":
            p = (it.get("predicate") or "").lower()
            v = (it.get("v") or "").lower()
            if "work" in ql and p in ("works_at", "work_at", "employer"):
                return v
            if ("city" in ql or "live" in ql) and p in ("lives_in", "live_in", "location"):
                return v
            if "name" in ql and p in ("name_is", "name"):
                return v

    # Chunk regex fallback
    for it in retrieved_items:
        if it.get("type") == "chunk":
            txt = (it.get("text") or "")
            if "work" in ql:
                m = WORK_AT_RE.search(txt)
                if m: return m.group(1).strip().rstrip(".").lower()
            if ("city" in ql or "live" in ql):
                m = LIVE_IN_RE.search(txt)
                if m: return m.group(1).strip().rstrip(".").lower()
            if "name" in ql:
                m = NAME_IS_RE.search(txt)
                if m: return m.group(1).strip().rstrip(".").lower()

    # Fallback: first token-ish
    return (retrieved_items[0].get("text", "unknown").split()[0].lower() if retrieved_items else "unknown")


def strings_from_index_snapshot(mem_name: str, memory_obj) -> List[str]:
    """
    Try to snapshot what was 'extracted/indexed'.
    - KeywordBaseline: chunks by text
    - VectorMemory: indexed chunk texts
    - GraphMemory: chunk texts + fact triples (subject/predicate/object)
    """
    out: List[str] = []
    try:
        if mem_name == "keyword_baseline":
            # Internal representation: memory.keyword_baseline.KeywordBaseline._docs
            for conv_id, chunks in memory_obj._docs.items():  # type: ignore
                out.extend([c.text for c in chunks])
        elif mem_name == "vector_memory":
            for conv_id, chunks in memory_obj._docs.items():  # type: ignore
                out.extend([c.text for c in chunks])
        elif mem_name == "graph_memory":
            # facts
            facts = memory_obj.facts.list_recent(limit=100000)  # type: ignore
            for f in facts:
                out.extend([f.subject, f.predicate, f.object])
            # chunk texts
            for conv_id, chunks in memory_obj._docs.items():  # type: ignore
                out.extend([c.text for c in chunks])
        else:
            # Unknown memory; do best effort if it exposes ._docs
            for conv_id, chunks in getattr(memory_obj, "_docs", {}).items():
                out.extend([getattr(c, "text", "") for c in chunks])
    except Exception:
        pass
    return out


def run_system(system: str, dataset: List[Dict[str, Any]], top_k: int, log_path: str | None, config_toggles: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """
    Returns (per_query_rows, latency_summary). Each row contains metrics for one eval item.
    """
    if system == "keyword":
        hooks = make_hooks("keyword_baseline", log_path=log_path, base_meta={"system": "keyword"})
        mem = make_keyword_baseline(hooks=hooks)
        mem_name = "keyword_baseline"
    elif system == "vector":
        hooks = make_hooks("vector_memory", log_path=log_path, base_meta={"system": "vector", **config_toggles})
        mem = make_vector_memory(hooks=hooks, rerank=bool(config_toggles.get("rerank", False)))
        # runtime config
        mem.configure(index_user_only=config_toggles.get("index_user_only", True),
                      dedup=config_toggles.get("dedup", True),
                      recency_weight=config_toggles.get("recency_weight", 0.10))
        mem_name = "vector_memory"
    elif system == "graph":
        hooks = make_hooks("graph_memory", log_path=log_path, base_meta={"system": "graph", **config_toggles})
        mem = make_graph_memory(hooks=hooks,
                                index_user_only=config_toggles.get("index_user_only", True),
                                confidence_threshold=config_toggles.get("confidence_threshold", 0.5))
        # weights toggle example
        if "weights" in config_toggles:
            a,b,g,d = config_toggles["weights"]
            mem.configure(weights=(a,b,g,d))
        mem_name = "graph_memory"
    elif system == "mem0":
        from utils.factory import make_hooks, make_mem0_memory
        hooks = make_hooks("mem0_memory", log_path=log_path, base_meta={"system": "mem0"})
        # Optionally pass a Mem0 config dict to pick vector store/graph store (see docs)
        mem = make_mem0_memory(hooks=hooks, mem0_config=None)
        mem_name = "mem0_memory"

    elif system == "li":  # LlamaIndex
        from utils.factory import make_hooks, make_llamaindex_memory
        hooks = make_hooks("llamaindex_memory", log_path=log_path, base_meta={"system": "llamaindex"})
        mem = make_llamaindex_memory(hooks=hooks)  # you can pass a llamaindex embed_model if you want
        mem_name = "llamaindex_memory"
    else:
        raise ValueError("system must be one of: keyword, vector, graph")

    rows: List[Dict[str, Any]] = []
    latencies: List[float] = []

    for conv in dataset:
        cid = conv["conversation_id"]
        for i, turn in enumerate(conv["turns"], start=1):
            mem.add_turn(cid, i, turn["role"], turn["content"])

        for ev in conv.get("eval", []):
            q = ev["question"]
            expected = ev["expected_answer"]
            relevant = ev.get("relevant_texts", [])
            subject_hint = ev.get("subject_hint", None)

            t0 = time.perf_counter()
            res = mem.retrieve(cid, q, top_k=top_k, subject_hint=subject_hint)  # subject_hint is accepted by GraphMemory; ignored otherwise
            latency_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(latency_ms)

            items = res.get("results", [])
            retrieved_texts: List[str] = []
            for it in items:
                # unify text across item types
                if it.get("type") == "fact":
                    retrieved_texts.append(f"{it.get('subject','')} {it.get('predicate','')} {it.get('object','')}")
                elif it.get("type") == "graph_edge":
                    retrieved_texts.append(f"{it.get('u','')} {it.get('predicate','')} {it.get('v','')}")
                else:
                    retrieved_texts.append(it.get("text", ""))

            pred = extract_answer(q, items)

            # snapshot what's indexed, for extraction-quality
            snapshot = strings_from_index_snapshot(mem_name, mem)

            row = {
                "conv_id": cid,
                "scenario": conv.get("scenario", "unknown"),
                "difficulty": conv.get("difficulty", "unknown"),
                "noise_tags": ",".join(conv.get("noise_tags", [])),
                "system": system,
                "top_k": top_k,
                "pred": pred,
                "gold": expected,
                "EM": float(pred.strip().lower() == expected.strip().lower()),
                "P@k": metric_retrieval_precision_k(retrieved_texts, relevant, top_k),
                "R@k": metric_retrieval_recall_k(retrieved_texts, relevant, top_k),
                "MRR": metric_mrr(retrieved_texts, relevant),
                "UpdateAcc": metric_update_accuracy(pred, expected, retrieved_texts, relevant) if conv.get("scenario") == "update" else None,
                "ExtractQ": metric_extraction_quality(snapshot, expected),
                "LatencyMS": float(latency_ms),
            }
            rows.append(row)

        # IMPORTANT: clear per-conv state if you evaluate each conv independently
        # For cross-session tests keep memory persistent; here we keep it persistent by design.

    latency_summary = metric_latency_ms(latencies)
    return rows, latency_summary


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def try_make_plots(summary_rows: List[Dict[str, Any]], out_dir: Path, title: str) -> None:
    """
    Optional bar charts if matplotlib is available.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # Bar by system x scenario: EM
    try:
        systems = sorted({r["system"] for r in summary_rows})
        scenarios = sorted({r["scenario"] for r in summary_rows})
        for metric in ("EM", "P@k", "R@k", "MRR"):
            for sc in scenarios:
                xs = []
                ys = []
                for sys in systems:
                    grp = [r for r in summary_rows if r["system"] == sys and r["scenario"] == sc]
                    val = sum(r.get(metric, 0.0) * r.get("count", 1) for r in grp) / max(1, sum(r.get("count", 1) for r in grp))
                    xs.append(sys)
                    ys.append(val)
                plt.figure()
                plt.bar(xs, ys)
                plt.title(f"{title} — {metric} by system on scenario: {sc}")
                plt.ylabel(metric)
                plt.xlabel("system")
                plt.tight_layout()
                plt.savefig(out_dir / f"{metric}_by_system_{sc}.png", dpi=150)
                plt.close()
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser(description="Evaluate memory systems on synthetic conversations.")
    ap.add_argument("--data", type=str, required=True, help="Path to conversations.jsonl generated by generate_dataset.py")
    ap.add_argument("--outdir", type=str, default="benchmark/results", help="Output directory for CSV/plots.")
    ap.add_argument("--top_k", type=int, default=5, help="Retriever top-k.")
    ap.add_argument("--compare", type=str, default="keyword,vector,graph,mem0,li",
                    help="Comma-separated systems: keyword,vector,graph,mem0,li")
    # A/B toggles for your system
    ap.add_argument("--vector_rerank", action="store_true", help="Enable LLM reranker in VectorMemory.")
    ap.add_argument("--no_vector_dedup", action="store_true", help="Disable dedup in VectorMemory.")
    ap.add_argument("--graph_low_conf", action="store_true", help="Lower confidence threshold in GraphMemory (0.3).")
    ap.add_argument("--weights", type=str, default="", help="Graph weights alpha,beta,gamma,delta (e.g. 0.5,0.3,0.1,0.1)")
    args = ap.parse_args()

    dataset = load_dataset(Path(args.data))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    systems = [s.strip() for s in args.compare.split(",") if s.strip()]

    all_rows: List[Dict[str, Any]] = []
    latency_rows: List[Dict[str, Any]] = []

    for sysname in systems:
        if sysname == "keyword":
            rows, lat = run_system(
                "keyword", dataset, top_k=args.top_k, log_path=str(outdir / "keyword_logs.jsonl"),
                config_toggles={}
            )
        elif sysname == "vector":
            rows, lat = run_system(
                "vector", dataset, top_k=args.top_k, log_path=str(outdir / "vector_logs.jsonl"),
                config_toggles={
                    "rerank": bool(args.vector_rerank),
                    "dedup": (not args.no_vector_dedup),
                }
            )
        elif sysname == "graph":
            weights = None
            if args.weights:
                try:
                    a,b,g,d = [float(x.strip()) for x in args.weights.split(",")]
                    weights = (a,b,g,d)
                except Exception:
                    weights = None
            rows, lat = run_system(
                "graph", dataset, top_k=args.top_k, log_path=str(outdir / "graph_logs.jsonl"),
                config_toggles={
                    "confidence_threshold": (0.3 if args.graph_low_conf else 0.5),
                    **({"weights": weights} if weights else {})
                }
            )
        else:
            raise ValueError(f"Unknown system: {sysname}")

        all_rows.extend(rows)
        latency_rows.append({"system": sysname, **lat})

    # Write per-query results
    write_csv(all_rows, outdir / "per_query_results.csv")

    # Summaries by system, scenario, difficulty
    # (average of metrics across groups)
    metric_keys = ["EM", "P@k", "R@k", "MRR", "ExtractQ"]
    by_system = aggregate_by_key(all_rows, "system", metric_keys)
    by_scenario = aggregate_by_key(all_rows, "scenario", ["EM", "P@k", "R@k", "MRR"])
    by_difficulty = aggregate_by_key(all_rows, "difficulty", ["EM", "P@k", "R@k", "MRR"])

    write_csv(by_system, outdir / "summary_by_system.csv")
    write_csv(by_scenario, outdir / "summary_by_scenario.csv")
    write_csv(by_difficulty, outdir / "summary_by_difficulty.csv")
    write_csv(latency_rows, outdir / "latency_summary.csv")

    # Optional quick plots
    try_make_plots(
        # Expand per-scenario bars: duplicate rows with 'count' to weight
        [{"system": r["system"], "scenario": r.get("scenario", "all"), "count": 1, **{m: r.get(m, 0.0) for m in metric_keys}} for r in all_rows],
        out_dir=(outdir / "plots"),
        title="Memory Benchmark"
    )

    print(f"\nWrote results to: {outdir}\n")
    print("Key files:")
    print(f"- {outdir/'per_query_results.csv'}")
    print(f"- {outdir/'summary_by_system.csv'}")
    print(f"- {outdir/'summary_by_scenario.csv'}")
    print(f"- {outdir/'summary_by_difficulty.csv'}")
    print(f"- {outdir/'latency_summary.csv'}")


if __name__ == "__main__":
    main()
