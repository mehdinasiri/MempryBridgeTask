#!/usr/bin/env python3
# repo/benchmark/generate_dataset.py
from __future__ import annotations

import argparse
import json
import random
import time
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from anyio import sleep
from loguru import logger

# --- OpenAI-compatible client ---
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore

# --- Project config ---
try:
    from utils.config import load_config
    CONFIG = load_config()
except Exception:
    from dotenv import load_dotenv
    import os
    load_dotenv()
    CONFIG = {
        "OPENAI_BASE_URL": os.getenv("OPENAI_BASE_URL", "").strip(),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", "").strip(),
        "MB_CHAT_MODEL": os.getenv("MB_CHAT_MODEL", "gpt-4o-mini").strip(),
    }

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


_PUNCT_END = re.compile(r"[ \t\r\n\.\!\?、。！？…]+$")


def _canon(s: str) -> str:
    """Canonicalize short evidence: lowercase, strip whitespace, trim trailing punctuation."""
    s = (s or "").strip().lower()
    return _PUNCT_END.sub("", s)


def _validate_example(obj: Dict[str, Any]) -> Dict[str, Any]:
    if "conversation_id" not in obj:
        raise ValueError("Missing conversation_id")

    # Ensure lists exist
    obj.setdefault("turns", [])
    obj.setdefault("eval", [])

    # ---------- normalize turns ----------
    turns = obj["turns"]
    if not isinstance(turns, list):
        raise ValueError("turns must be a list")

    norm_turns: List[Dict[str, str]] = []
    for idx, t in enumerate(turns, start=1):
        role = (t.get("role") or "").strip().lower()
        content = (t.get("content") or "").strip()
        if role not in ("user", "assistant"):
            raise ValueError(f"Invalid role '{role}' at turn {idx}")
        norm_turns.append({"role": role, "content": content})
    obj["turns"] = norm_turns

    # ---------- normalize eval ----------
    norm_evals: List[Dict[str, Any]] = []
    for e in obj["eval"]:
        item = {
            "question": (e.get("question") or "").strip().lower(),
            "expected_answer": (e.get("expected_answer") or "").strip().lower(),
            "relevant_texts": [_canon(s) for s in (e.get("relevant_texts") or [])],
        }
        if e.get("subject_hint"):
            item["subject_hint"] = (e["subject_hint"] or "").strip()
        norm_evals.append(item)
    obj["eval"] = norm_evals

    # ---------- normalize labels ----------
    obj["difficulty"] = (obj.get("difficulty") or "").strip().lower()
    obj["scenario"] = (obj.get("scenario") or "").strip().lower()
    if not isinstance(obj.get("noise_tags"), list):
        obj["noise_tags"] = []

    # ---------- align eval question to final user question if needed ----------
    if obj["eval"]:
        q = obj["eval"][0]["question"]
        user_utts = [t["content"] for t in obj["turns"] if t["role"] == "user"]
        user_utts_l = [u.strip().lower() for u in user_utts]

        # exact contains match first (case-insensitive)
        found_exact = any(q == u for u in user_utts_l) or any(q in u for u in user_utts_l)

        if not found_exact:
            # try to pick the last user question as gold
            last_user_q = None
            for u in reversed(user_utts):
                u_stripped = u.strip()
                if u_stripped.endswith("?"):
                    last_user_q = u_stripped.lower()
                    break
            if last_user_q:
                logger.warning(
                    f"[validate] Adjusting eval.question to last user question: "
                    f"'{obj['eval'][0]['question']}' -> '{last_user_q}' "
                    f"(conv={obj['conversation_id']})"
                )
                obj["eval"][0]["question"] = last_user_q
            else:
                # still not found, raise—this keeps bad items out
                raise ValueError(
                    f"Eval question '{q}' not found in user turns for {obj['conversation_id']}"
                )
    else:
        raise ValueError("Model did not return eval[]")

    return obj


def _llm_client() -> OpenAI:
    if OpenAI is None:
        raise RuntimeError("openai client not available. Install openai>=1.0 and set creds.")
    return OpenAI(
        base_url=(CONFIG.get("OPENAI_BASE_URL") or None),
        api_key=CONFIG.get("OPENAI_API_KEY", ""),
    )


def _chat_json(messages: List[Dict[str, str]], temperature: float, max_retries: int = 3) -> Dict[str, Any]:
    client = _llm_client()
    last_err = None
    for _ in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=CONFIG.get("MB_CHAT_MODEL", "gpt-4o-mini"),
                messages=messages,
                temperature=temperature,
                response_format={"type": "json_object"},
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
# Prompt templates
# =========================

SYS = (
    "You are a data generator for evaluating conversational memory systems. "
    "Your job is to create synthetic multi-turn dialogues that rigorously test memory. "
    "ALWAYS return STRICT JSON only, no prose, matching the requested schema. "
    "Never include commentary outside JSON."
)


SCENARIO_BRIEF = {
    "simple": (
        "SIMPLE FACT STORAGE & RECALL.\n"
        "- The user states a personal fact early in the chat.\n"
        "- Later, the user asks a question that requires recalling that exact fact.\n"
    ),
    "update": (
        "FACT UPDATE / CORRECTION.\n"
        "- The user states a fact, then later corrects or updates it.\n"
        "- The final question asks for the CURRENT value (most recent truth).\n"
    ),
    "multihop": (
        "MULTI-HOP REASONING.\n"
        "- The user gives two+ related facts across turns.\n"
        "- The final question requires combining them.\n"
        "- If two fields must be returned, join them exactly with \"; \" (semicolon+space), not words.\n"
    ),
}


DIFFICULTY_GUIDE = {
    "easy": (
        "EASY: Minimal noise/distraction; direct statements; no typos; clear phrasing; "
        "short chat (4–6 turns)."
    ),
    "moderate": (
        "MODERATE: Add realistic noise (chit-chat, typos, minor coreference); "
        "medium length (4–8 turns); answer still unambiguous."
    ),
    "complex": (
        "COMPLEX: Heavier noise (multi-topic chatter, emojis, multilingual snippets), "
        "stronger coreference, long gaps between fact & question; longer chat (8–10 turns)."
    ),
}

JSON_SCHEMA_GUIDE = (
    "Return JSON with keys:\n"
    "{\n"
    '  "conversation_id": str,\n'
    '  "scenario": "simple" | "update" | "multihop",\n'
    '  "difficulty": "easy" | "moderate" | "complex",\n'
    '  "noise_tags": [str, ...],\n'
    '  "turns": [{"role":"user|assistant","content": str}, ...],\n'
    '  "eval": [\n'
    "    {\n"
    '      "question": str,\n'
    '      "expected_answer": str,  // short phrase only per rules\n'
    '      "relevant_texts": [str, ...],\n'
    '      "subject_hint": str | null\n'
    "    }\n"
    "  ]\n"
    "}\n"
)

def scenario_user_prompt(scenario: str, difficulty: str, conv_id: str, seed: int) -> str:
    return (
        f"SCENARIO: {scenario.upper()}\n\n"
        f"{SCENARIO_BRIEF[scenario]}\n"
        f"DIFFICULTY: {difficulty.upper()}\n"
        f"{DIFFICULTY_GUIDE[difficulty]}\n\n"
        "Goal\n"
        "Create ONE conversation and ONE eval item.\n"
        "The eval item verifies a SINGLE factual point from the conversation.\n\n"
        "Hard constraints (must all be satisfied)\n"
        "1) Exactly ONE eval item for the conversation.\n"
        "2) eval.question = the FINAL user message, asking about EXACTLY ONE fact only.\n"
        "   - No compound questions, no lists, no comparisons.\n"
        "   - Disallow conjunctions/joins: 'and', 'or', 'as well as', '&', '/'.\n"
        "   - Disallow commas/semicolons/colons in the question.\n"
        "   - End with a single '?' and be ≤ 120 characters.\n"
        "3) eval.expected_answer = a SHORT PHRASE only:\n"
        "   - all lowercase, no leading/trailing spaces, no punctuation other than internal spaces or hyphens\n"
        "   - not a full sentence; must not contain verbs like 'is/are/was/were/do/does/did/can/should'\n"
        "   - no trailing period\n"
        "4) eval.relevant_texts = 1–3 snippets supporting the answer:\n"
        "   - each snippet lowercase and trimmed, no trailing punctuation\n"
        "   - quote exact or near-exact spans from the conversation (no new facts)\n"
        "   - 5–140 characters each\n"
        "5) conversation/eval fields MUST match exactly:\n"
        f"   - conversation_id == '{conv_id}'\n"
        f"   - scenario == '{scenario}'\n"
        f"   - difficulty == '{difficulty}'\n"
        "6) Output STRICT JSON ONLY. No backticks, no commentary, no XML/YAML.\n"
        "7) ASCII characters only.\n\n"
        "Guidance\n"
        "- If a natural question would require two facts, rewrite it to focus on one dimension only.\n"
        "  Example: instead of 'what do giant pandas eat and where are they from?', ask 'what do giant pandas eat?'\n"
        "- Prefer concrete, unambiguous facts that can be justified by 1–2 short snippets.\n"
        "- Keep conversation realistic but concise; avoid repeating the exact answer verbatim multiple times.\n\n"
        "Forbidden patterns (reject/avoid and rewrite BEFORE outputting JSON)\n"
        "- Questions containing: ' and ', ' or ', ',', ';', ':', '/', '&', ' as well as '\n"
        "- expected_answer with uppercase letters, punctuation at the end, or being a full sentence.\n"
        "- relevant_texts with trailing punctuation or text not present/implied in the conversation.\n\n"
        "Self-check BEFORE emitting JSON (all must be true)\n"
        "- [Q1] Is the final question single-fact and ends with '?' with no commas/joins?\n"
        "- [A1] Is expected_answer a short, lowercase phrase (no sentence, no trailing punctuation, no verbs like 'is/are')?\n"
        "- [E1] Do 1–3 relevant_texts justify A1, are lowercase, trimmed, and have no trailing punctuation?\n"
        "- [M1] Do conversation_id/scenario/difficulty EXACTLY match the specified values?\n"
        "- If any check fails, fix the conversation/question/answer and re-check.\n\n"
        f"Seed: {seed}\n\n"
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
            "model": CONFIG.get("MB_CHAT_MODEL", "gpt-4o-mini"),
        })

    raw = _chat_json(messages, temperature=temperature)
    obj = _validate_example(raw)

    # enforce keys
    obj["conversation_id"] = conv_id
    obj["scenario"] = scenario
    obj["difficulty"] = difficulty
    if "noise_tags" not in obj or not isinstance(obj["noise_tags"], list):
        obj["noise_tags"] = []

    # ensure exactly one eval
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
    per_scenario: Dict[str, Dict[str, int]],
    temps: Dict[str, float] = None,
) -> None:
    rng = random.Random(seed)
    temps = temps or {"easy": 0.6, "moderate": 0.8, "complex": 1.0}

    total = sum(per_scenario[s][d] for s in per_scenario for d in per_scenario[s])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Generating {total} conversations → {out_path}")

    generated = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for scenario in ("simple", "update", "multihop"):
            if scenario not in per_scenario:
                continue
            for difficulty in ("easy", "moderate", "complex"):
                n = int(per_scenario[scenario].get(difficulty, 0))
                for i in range(n):
                    conv_id = f"{scenario.upper()}_{difficulty.upper()}_{seed}_{i:03d}"
                    while True:
                        try:
                            obj = generate_example(
                                scenario=scenario,
                                difficulty=difficulty,
                                conv_id=conv_id,
                                seed=rng.randrange(10**9),
                                temperature=temps.get(difficulty, 0.7),
                                promptlog=promptlog_path,
                            )
                            break
                        except Exception as e:
                            print(e)
                            sleep(10)
                    fh.write(json.dumps(obj, ensure_ascii=False) + "\n")

                    generated += 1
                    logger.info(f"Progress: {generated}/{total} conversations generated")

    logger.info("Done.")

# =========================
# CLI
# =========================

def main():
    ap = argparse.ArgumentParser(description="LLM-driven synthetic dataset generator for memory benchmarking (JSONL).")
    ap.add_argument("--seed", type=int, default=7, help="Random seed for IDs/variety.")
    ap.add_argument("--output", type=str, default="datasets/conversations.jsonl", help="Output JSONL path.")
    ap.add_argument("--promptlog", type=str, default="datasets/promptlog.jsonl", help="Prompt log JSONL path.")
    ap.add_argument("--simple", type=str, default="6,6,6", help="Counts easy,moderate,complex for SIMPLE.")
    ap.add_argument("--update", type=str, default="6,6,6", help="Counts easy,moderate,complex for UPDATE.")
    ap.add_argument("--multihop", type=str, default="6,6,6", help="Counts easy,moderate,complex for MULTIHOP.")
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

    out_path = Path(args.output)
    promptlog_path = Path(args.promptlog) if args.promptlog else None

    if not CONFIG.get("OPENAI_API_KEY"):
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
