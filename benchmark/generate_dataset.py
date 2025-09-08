#!/usr/bin/env python3
# repo/benchmark/generate_dataset.py
from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# --- OpenAI-compatible client (Gemini via your proxy is fine) ---
try:
    from openai import OpenAI
except Exception as e:
    OpenAI = None  # type: ignore

# Project config (for base_url, api_key, model names)
try:
    from utils.config import SETTINGS
except Exception:
    class _S:
        base_url = ""
        api_key = ""
        chat_model = "gpt-4o-mini"
    SETTINGS = _S()  # type: ignore


# =========================
# Schema & helpers
# =========================

@dataclass
class EvalItem:
    question: str
    expected_answer: str
    relevant_texts: List[str]
    subject_hint: Optional[str] = None

def _strict_json_extract(txt: str) -> Dict[str, Any]:
    """Best-effort to extract a single JSON object from a model reply."""
    start = txt.find("{")
    end = txt.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found in model response.")
    return json.loads(txt[start:end+1])

def _validate_example(obj: Dict[str, Any]) -> Dict[str, Any]:
    # Minimal validation + normalization for downstream scripts
    if "conversation_id" not in obj:
        raise ValueError("Missing conversation_id")
    obj.setdefault("turns", [])
    obj.setdefault("eval", [])
    for t in obj["turns"]:
        t["role"] = (t.get("role") or "").strip().lower()
        assert t["role"] in ("user", "assistant")
        t["content"] = (t.get("content") or "").strip()
    # normalize eval items
    norm_evals = []
    for e in obj["eval"]:
        item = {
            "question": (e.get("question") or "").strip(),
            "expected_answer": (e.get("expected_answer") or "").strip().lower(),
            "relevant_texts": [s.strip().lower() for s in (e.get("relevant_texts") or [])],
        }
        if e.get("subject_hint"):
            item["subject_hint"] = (e["subject_hint"] or "").strip()
        norm_evals.append(item)
    obj["eval"] = norm_evals
    # difficulty/scenario/noise labels
    obj["difficulty"] = (obj.get("difficulty") or "").strip().lower()
    obj["scenario"] = (obj.get("scenario") or "").strip().lower()
    obj["noise_tags"] = obj.get("noise_tags", [])
    return obj

def _llm_client() -> OpenAI:
    if OpenAI is None:
        raise RuntimeError("openai client not available. Install openai>=1.0 and set creds.")
    return OpenAI(base_url=(SETTINGS.base_url or None), api_key=SETTINGS.api_key)

def _chat_json(messages: List[Dict[str, str]], temperature: float, max_retries: int = 3) -> Dict[str, Any]:
    client = _llm_client()
    last_err = None
    for _ in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=SETTINGS.chat_model,
                messages=messages,
                temperature=temperature,
                response_format={"type": "json_object"},  # many gateways support this
            )
            content = resp.choices[0].message.content or ""
            return _strict_json_extract(content)
        except Exception as e:
            last_err = e
            time.sleep(0.8)
    raise last_err if last_err else RuntimeError("Unknown LLM error")

def _promptlog_write(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")

# =========================
# Prompt templates (no hardcoded entities)
# =========================

SYS = (
    "You are a data generator for evaluating conversational memory systems. "
    "Your job is to create synthetic multi-turn dialogues that rigorously test memory. "
    "ALWAYS return STRICT JSON only, no prose, matching the requested schema."
)

SCENARIO_BRIEF = {
    "simple": (
        "SIMPLE FACT STORAGE & RECALL.\n"
        "- The user states a personal fact early in the chat (e.g., name, city, favorite, employer).\n"
        "- Later, the user asks a question that requires recalling that exact fact.\n"
    ),
    "update": (
        "FACT UPDATE / CORRECTION.\n"
        "- The user states a fact, then later corrects or updates it (new employer, new city, changed preference).\n"
        "- The final question asks for the CURRENT value (most recent truth), not the outdated one.\n"
    ),
    "multihop": (
        "MULTI-HOP REASONING.\n"
        "- The user gives two+ related facts across turns.\n"
        "- The final question requires combining them (e.g., manager → manager's employer; city → country).\n"
    ),
}

DIFFICULTY_GUIDE = {
    "easy": (
        "EASY: Minimal noise/distraction; direct statements; no typos; clear phrasing; "
        "short chat (6–10 turns)."
    ),
    "moderate": (
        "MODERATE: Add realistic noise (chit-chat, small typos, minor coreference like 'they', "
        "light paraphrase); medium length (10–16 turns); keep the answer still unambiguous."
    ),
    "complex": (
        "COMPLEX: Heavier noise (multi-topic chatter, emojis, occasional code-switching "
        "or multilingual snippets), stronger coreference, possible long gaps between fact & question; "
        "may include cross-session separator lines like '(end of session)/(new session)'; longer chat (16–24 turns)."
    ),
}

JSON_SCHEMA_GUIDE = (
    "Return JSON with keys:\n"
    '{\n'
    '  "conversation_id": str,\n'
    '  "scenario": "simple" | "update" | "multihop",\n'
    '  "difficulty": "easy" | "moderate" | "complex",\n'
    '  "noise_tags": [str, ...],  // e.g., ["typos","emoji","multilingual","coreference","cross_session"]\n'
    '  "turns": [{"role":"user|assistant","content": str}, ...],\n'
    '  "eval": [\n'
    '    {\n'
    '      "question": str,\n'
    '      "expected_answer": str,    // lowercase canonical\n'
    '      "relevant_texts": [str,...], // lowercase snippets that justify the answer\n'
    '      "subject_hint": str | null // optional subject string helpful for graph retrieval\n'
    '    }\n'
    '  ]\n'
    '}\n'
    "Rules:\n"
    "- The answer MUST be recoverable from the conversation content.\n"
    "- Use realistic, varied language; do NOT reuse canned entities.\n"
    "- Keep expected_answer lowercase; relevant_texts lowercase.\n"
    "- The final user question should appear as a user turn and also in eval[].\n"
)

def scenario_user_prompt(scenario: str, difficulty: str, conv_id: str, seed: int) -> str:
    return (
        f"SCENARIO: {scenario.upper()}\n\n"
        f"{SCENARIO_BRIEF[scenario]}\n"
        f"DIFFICULTY: {difficulty.upper()}\n"
        f"{DIFFICULTY_GUIDE[difficulty]}\n\n"
        "Noise guidance by difficulty:\n"
        "- easy: no typos; minimal filler; no multilingual; no emoji; no cross-session.\n"
        "- moderate: add some filler/typos/coreference; minor paraphrase.\n"
        "- complex: more filler; stronger coreference; emojis; occasional multilingual words; "
        " include a clear cross-session separator if natural.\n\n"
        "Diversity requirements:\n"
        "- Use names, places, employers, hobbies, etc. that you invent on the fly; be varied.\n"
        "- Avoid brand repetition across samples; do not use the exact same strings repeatedly.\n"
        "- Keep content safe-for-work.\n\n"
        "Evaluation target:\n"
        "- Provide exactly ONE eval item per conversation.\n"
        "- The question MUST directly match the final user question turn.\n"
        "- expected_answer MUST be lowercased and match a canonical form present or derivable from the dialogue.\n"
        "- relevant_texts MUST include 1–3 lowercase snippets that justify the answer.\n"
        "- subject_hint is optional but helpful (e.g., a person name, 'user').\n\n"
        "Output constraints:\n"
        f"- conversation_id MUST be '{conv_id}'.\n"
        f"- scenario MUST be '{scenario}'.\n"
        f"- difficulty MUST be '{difficulty}'.\n"
        "- turns MUST alternate naturally and be coherent; length by difficulty guide.\n"
        "- STRICT JSON ONLY. No commentary.\n\n"
        f"Seed (for diversity, not for randomness guarantees): {seed}\n\n"
        + JSON_SCHEMA_GUIDE
    )

# =========================
# Generation driver
# =========================

def generate_example(
    scenario: str,
    difficulty: str,
    conv_id: str,
    seed: int,
    temperature: float,
    promptlog: Optional[Path],
) -> Dict[str, Any]:
    system_msg = {"role": "system", "content": SYS}
    user_msg = {"role": "user", "content": scenario_user_prompt(scenario, difficulty, conv_id, seed)}
    messages = [system_msg, user_msg]

    if promptlog:
        _promptlog_write(promptlog, {
            "conversation_id": conv_id,
            "scenario": scenario,
            "difficulty": difficulty,
            "temperature": temperature,
            "system": SYS,
            "user": user_msg["content"],
            "model": getattr(SETTINGS, "chat_model", "unknown")
        })

    raw = _chat_json(messages, temperature=temperature)
    obj = _validate_example(raw)
    # final safety: ensure conv_id/difficulty/scenario match our request
    obj["conversation_id"] = conv_id
    obj["scenario"] = scenario
    obj["difficulty"] = difficulty
    # ensure noise_tags present
    if "noise_tags" not in obj or not isinstance(obj["noise_tags"], list):
        obj["noise_tags"] = []
    # ensure exactly one eval item
    if not obj["eval"]:
        raise ValueError("Model did not return eval[]")
    if len(obj["eval"]) > 1:
        obj["eval"] = [obj["eval"][0]]
    return obj


def generate_dataset(
    *,
    seed: int,
    out_path: Path,
    promptlog_path: Optional[Path],
    per_scenario: Dict[str, Dict[str, int]],  # e.g., {"simple":{"easy":10,"moderate":10,"complex":10}, ...}
    temps: Dict[str, float] = None,           # temperature per difficulty
) -> None:
    rng = random.Random(seed)
    temps = temps or {"easy": 0.6, "moderate": 0.8, "complex": 1.0}

    total = sum(per_scenario[s][d] for s in per_scenario for d in per_scenario[s])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Generating {total} conversations → {out_path}")

    with out_path.open("w", encoding="utf-8") as fh:
        for scenario in ("simple", "update", "multihop"):
            if scenario not in per_scenario:
                continue
            for difficulty in ("easy", "moderate", "complex"):
                n = int(per_scenario[scenario].get(difficulty, 0))
                for i in range(n):
                    conv_id = f"{scenario.upper()}_{difficulty.upper()}_{seed}_{i:03d}"
                    obj = generate_example(
                        scenario=scenario,
                        difficulty=difficulty,
                        conv_id=conv_id,
                        seed=rng.randrange(10**9),
                        temperature=temps.get(difficulty, 0.7),
                        promptlog=promptlog_path,
                    )
                    fh.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print("Done.")

# =========================
# CLI
# =========================

def main():
    ap = argparse.ArgumentParser(description="LLM-driven synthetic dataset generator for memory benchmarking (JSONL).")
    ap.add_argument("--seed", type=int, default=7, help="Random seed for IDs/variety (not strict).")
    ap.add_argument("--out", type=str, default="benchmark/datasets/conversations.json", help="Output JSONL path.")
    ap.add_argument("--promptlog", type=str, default="benchmark/datasets/promptlog.jsonl", help="Prompt log JSONL path.")
    # Per-scenario counts (easy/moderate/complex)
    ap.add_argument("--simple", type=str, default="10,10,10", help="Counts easy,moderate,complex for SIMPLE.")
    ap.add_argument("--update", type=str, default="10,10,10", help="Counts easy,moderate,complex for UPDATE.")
    ap.add_argument("--multihop", type=str, default="10,10,10", help="Counts easy,moderate,complex for MULTIHOP.")
    # Difficulty temperatures
    ap.add_argument("--temps", type=str, default="0.6,0.8,1.0", help="Temperatures for easy,moderate,complex.")
    args = ap.parse_args()

    def parse_counts(s: str) -> Dict[str, int]:
        a, b, c = [int(x.strip()) for x in s.split(",")]
        return {"easy": a, "moderate": b, "complex": c}

    def parse_temps(s: str) -> Dict[str, float]:
        a, b, c = [float(x.strip()) for x in s.split(",")]
        return {"easy": a, "moderate": b, "complex": c}

    per = {
        "simple": parse_counts(args.simple),
        "update": parse_counts(args.update),
        "multihop": parse_counts(args.multihop),
    }
    temps = parse_temps(args.temps)

    out_path = Path(args.out)
    promptlog_path = Path(args.promptlog) if args.promptlog else None

    # Sanity: ensure we have credentials
    if not getattr(SETTINGS, "api_key", ""):
        raise RuntimeError("OPENAI_API_KEY missing. Set via .env")

    generate_dataset(
        seed=args.seed,
        out_path=out_path,
        promptlog_path=promptlog_path,
        per_scenario=per,
        temps=temps,
    )

if __name__ == "__main__":
    main()
