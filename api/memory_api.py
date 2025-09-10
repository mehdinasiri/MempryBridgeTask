from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any, List, Optional

# Import memory classes
from memory.graph_memory import GraphMemory
from memory.keyword_baseline import KeywordBaseline
from memory.third_party.llamaindex_adapter import LlamaIndexMemory
from memory.third_party.mem0_adapter import Mem0Memory
from memory.vector_memory import VectorMemory

import os
from utils.config import load_config

app = FastAPI(title="Unified Memory API")

# ---------- Init memories ----------
cfg = load_config()

graph_mem = GraphMemory(restrict_to_conv=True, max_hops=2, enable_query_fact_extraction=True)
keyword_mem = KeywordBaseline(name="keyword_baseline_api")
llamaindex_mem = LlamaIndexMemory(
    name="api_llamaindex",
    collection=os.getenv("MB_COLLECTION", "mem0_benchmark_dev"),
    chat_model=os.getenv("MB_CHAT_MODEL", "gemini-flash"),
    embed_model=os.getenv("MB_EMBED_MODEL", "gemini-embedding"),
    base_url=os.getenv("OPENAI_BASE_URL", ""),
    api_key=os.getenv("OPENAI_API_KEY", ""),
    restrict_to_conv=True,
)
mem0_mem = Mem0Memory(
    name="api_mem0",
    collection=os.getenv("MB_COLLECTION", "mem0_default"),
    chat_model=os.getenv("MB_CHAT_MODEL", "gemini-flash"),
    embed_model=os.getenv("MB_EMBED_MODEL", "gemini-embedding"),
    base_url=os.getenv("OPENAI_BASE_URL", ""),
    api_key=os.getenv("OPENAI_API_KEY", ""),
    restrict_to_conv=True,
)
vector_mem = VectorMemory(
    name="api_vm",
    index_backend="chroma",
    collection_or_table="api_facts",
    persist_path=".memdb/chroma_api",
    restrict_to_conv=True,
)

# ---------- Schemas ----------
class InsertRequest(BaseModel):
    conv_id: str
    turn_id: int
    role: str
    content: str

class RetrieveRequest(BaseModel):
    conv_id: str
    query: str
    top_k: Optional[int] = 3

# ---------- Helper ----------
def insert_turn(mem, conv_id: str, turn_id: int, role: str, content: str):
    mem.add_turn(conv_id, turn_id, role, content)
    return {"status": "ok", "conv_id": conv_id, "turn_id": turn_id}

def retrieve_facts(mem, conv_id: str, query: str, top_k: int):
    res = mem.retrieve(conv_id, query, top_k=top_k)
    return res

# ---------- GraphMemory ----------
@app.post("/graph/insert")
def graph_insert(req: InsertRequest):
    return insert_turn(graph_mem, req.conv_id, req.turn_id, req.role, req.content)

@app.post("/graph/retrieve")
def graph_retrieve(req: RetrieveRequest):
    return retrieve_facts(graph_mem, req.conv_id, req.query, req.top_k)

# ---------- KeywordBaseline ----------
@app.post("/keyword/insert")
def keyword_insert(req: InsertRequest):
    return insert_turn(keyword_mem, req.conv_id, req.turn_id, req.role, req.content)

@app.post("/keyword/retrieve")
def keyword_retrieve(req: RetrieveRequest):
    return retrieve_facts(keyword_mem, req.conv_id, req.query, req.top_k)

# ---------- LlamaIndexMemory ----------
@app.post("/llama/insert")
def llama_insert(req: InsertRequest):
    return insert_turn(llamaindex_mem, req.conv_id, req.turn_id, req.role, req.content)

@app.post("/llama/retrieve")
def llama_retrieve(req: RetrieveRequest):
    return retrieve_facts(llamaindex_mem, req.conv_id, req.query, req.top_k)

# ---------- Mem0Memory ----------
@app.post("/mem0/insert")
def mem0_insert(req: InsertRequest):
    return insert_turn(mem0_mem, req.conv_id, req.turn_id, req.role, req.content)

@app.post("/mem0/retrieve")
def mem0_retrieve(req: RetrieveRequest):
    return retrieve_facts(mem0_mem, req.conv_id, req.query, req.top_k)

# ---------- VectorMemory ----------
@app.post("/vector/insert")
def vector_insert(req: InsertRequest):
    return insert_turn(vector_mem, req.conv_id, req.turn_id, req.role, req.content)

@app.post("/vector/retrieve")
def vector_retrieve(req: RetrieveRequest):
    return retrieve_facts(vector_mem, req.conv_id, req.query, req.top_k)
