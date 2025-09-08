# demo_complex_graph_memory.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any, List

from loguru import logger
from memory.graph_memory import GraphMemory
from pyvis.network import Network

def render_conv_graph_pyvis(mem, conv_id: str, out_dir: str = "viz") -> str:
    if not hasattr(mem, "_G") or conv_id not in mem._G:
        print(f"[viz] No graph for conversation: {conv_id}")
        return ""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    html_path = str(out / f"graph_{conv_id}.html")

    G = mem._G[conv_id]

    net = Network(height="92vh", width="100%", directed=True, bgcolor="#ffffff")
    # physics & style must be JSON, not JS:
    net.set_options("""
    {
      "edges": { "font": { "size": 12, "align": "middle" } },
      "nodes": { "font": { "size": 14 } },
      "physics": {
        "enabled": true,
        "stabilization": { "iterations": 250 },
        "solver": "forceAtlas2Based",
        "forceAtlas2Based": {
          "gravitationalConstant": -35,
          "centralGravity": 0.005,
          "springConstant": 0.08
        }
      },
      "interaction": { "hover": true, "tooltipDelay": 120 }
    }
    """)

    # nodes
    for n in G.nodes():
        net.add_node(n, label=n, title=n, shape="dot", size=16)

    # edges with hover tooltips
    for u, v, k, data in G.edges(keys=True, data=True):
        pred = data.get("predicate", "")
        conf = data.get("confidence", 0.0)
        ev   = (data.get("evidence") or "").strip()
        title = f"<b>{u}</b> — <i>{pred}</i> → <b>{v}</b><br/>conf: {conf:.2f}"
        if ev:
            title += f"<br/>{ev}"
        net.add_edge(u, v, label=pred, title=title, arrows="to", smooth=True)

    net.write_html(html_path, open_browser=False, notebook=False)
    print(f"[viz] wrote: {html_path}")
    return html_path

def print_results(question: str, results: List[Dict[str, Any]]) -> None:
    print(f"\nQ: {question}")
    if not results:
        print("   [NO RESULTS]")
        return
    print("   Top results:")
    for i, r in enumerate(results, start=1):
        subj = r.get("subject")
        pred = r.get("predicate")
        obj = r.get("object")
        score = r.get("score", 0.0)
        ev = r.get("text") or ""
        print(f"   #{i:02d}: ({subj} | {pred} | {obj})")
        print(f"        score={score:.3f}  evidence='{ev}'")


def stream_conversation(mem: GraphMemory, convo: Dict[str, Any]) -> None:
    conv_id = convo["conversation_id"]
    print(f"\n=== Conversation {conv_id} ===")

    for i, t in enumerate(convo["turns"], start=1):
        mem.add_turn(conv_id, i, t["role"], t["content"])

    for q in convo.get("queries", []):
        res = mem.retrieve(conv_id, q, top_k=5)
        print_results(q, res["results"])

    html_path = render_conv_graph_pyvis(mem, conv_id, out_dir="viz")
    if html_path:
        print(f"\n   → Graph saved: {html_path}")


def main():
    mem = GraphMemory(
        restrict_to_conv=True,
        max_hops=2,
        enable_query_fact_extraction=True,
    )

    CONVOS: List[Dict[str, Any]] = [
        {
            "conversation_id": "MULTI_HOP_COREF_001",
            "turns": [
                {"role": "user", "content": "My colleague, Liam Chen, is a brilliant data scientist."},
                {"role": "assistant", "content": "Nice!"},
                {"role": "user", "content": "He moved jobs. He's at Quantum Innovations now."},
                {"role": "assistant", "content": "Got it."},
                {"role": "user", "content": "I think their main office is somewhere in Europe."},
                {"role": "assistant", "content": "Okay."},
                {"role": "user", "content": "(new session) Okay, I remember: it's Stockholm. Definitely Stockholm."},
            ],
            "queries": [
                "Where does Liam work now?",
                "What city is Liam's employer based in?",
                "What continent is that city in?",
            ],
        },
        {
            "conversation_id": "UPDATES_SUPERSEDES_002",
            "turns": [
                {"role": "user", "content": "My name is Alice. I work at OpenAI."},
                {"role": "assistant", "content": "Hello Alice!"},
                {"role": "user", "content": "Actually, I switched companies. I work at Anthropic now."},
                {"role": "assistant", "content": "Updated."},
                {"role": "user", "content": "Anthropic is headquartered in San Francisco."},
            ],
            "queries": [
                "Where does Alice work?",
                "What city is Alice's employer headquartered in?",
            ],
        },
        {
            "conversation_id": "UNCERTAINTY_COMPETING_003",
            "turns": [
                {"role": "user", "content": "Bob works at Nova Labs."},
                {"role": "assistant", "content": "Noted."},
                {"role": "user", "content": "Nova Labs is focused on robotics."},
                {"role": "assistant", "content": "Interesting."},
                {"role": "user", "content": "I heard they might move their HQ to Berlin soon, not sure though."},
                {"role": "assistant", "content": "Okay."},
                {"role": "user", "content": "Correction: their main office is actually in Munich."},
            ],
            "queries": [
                "What does Nova Labs focus on?",
                "Where is Nova Labs' main office?",
                "Which city is Bob's employer based in?",
            ],
        },
        {
            "conversation_id": "LONGER_CHAIN_004",
            "turns": [
                {"role": "user", "content": "Zara founded Helix Bio."},
                {"role": "assistant", "content": "Okay."},
                {"role": "user", "content": "Helix Bio acquired MicroCore."},
                {"role": "assistant", "content": "Noted."},
                {"role": "user", "content": "MicroCore is based in Cambridge."},
                {"role": "assistant", "content": "Thanks."},
                {"role": "user", "content": "Zara later became the CTO at Helix Bio."},
                {"role": "assistant", "content": "Updated."},
                {"role": "user", "content": "Helix Bio’s European HQ is in Zurich."},
                {"role": "assistant", "content": "Okay."},
                {"role": "user", "content": "Also, unrelated: I love espresso."},
            ],
            "queries": [
                "Where is the European HQ of the company Zara is CTO at?",
                "Which city is the subsidiary of Zara's company based in?",
            ],
        },
    ]

    for convo in CONVOS:
        stream_conversation(mem, convo)


if __name__ == "__main__":
    main()