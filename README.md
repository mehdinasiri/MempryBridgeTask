# MemoryBridge Task

## Introduction

This project evaluates three types of memory systems—keyword, vector, and graph—to determine their effectiveness in remembering and answering questions from conversations. The provided demo scripts and API allow you to explore and benchmark these memory approaches in practical scenarios.
## Prerequisites

- Python 3.8+
- `pip`
- `docker` and `docker compose` (for API deployment)

## Setup

### 1. Configure Environment Variables

Copy the example environment file and export variables:

```bash
cp .env.example .env
export $(cat .env | xargs)
```

Edit `.env` as needed for your environment.

### 2. Install Python Dependencies

Install required packages:

```bash
pip install -r requirements.txt
```

## Running Demo Files

To run a demo script, use one of the following commands:

```bash
python -m demo.demo_graph_memory
python -m demo.demo_keyword_baseline
python -m demo.demo_llamaindex_memory
python -m demo.demo_mem0_memory
python -m demo.demo_vector_memory
```

## Running Dataset Generation and Evaluation

The `benchmark` folder provides tools to create datasets and evaluate different memory systems.

### 1. Generate a Dataset

You can generate synthetic conversation datasets for benchmarking:

```bash
python benchmark/generate_dataset.py \
  --output benchmark/datasets/conversations.jsonl \
  --promptlog benchmark/datasets/promptlog.jsonl \
  --simple 5,5,5 \
  --update 6,6,6 \
  --multihop 7,7,7
```

This will:
* `--simple 5,5,5` → generate **5 easy, 5 moderate, 5 complex** SIMPLE tasks
* `--update 6,6,6` → generate **6 easy, 6 moderate, 6 complex** UPDATE tasks
* `--multihop 7,7,7` → generate **7 easy, 7 moderate, 7 complex** MULTIHOP tasks


### 2. Run Evaluation

```bash
python benchmark/evaluate.py \
  --data benchmark/datasets/conversations.jsonl \
  --compare keyword,vector,graph \
  --top_k 5 \
  --index_backend chroma \
  --collection_or_table memorybridge_facts \
  --vector_db_path .memdb/vector_chroma \
  --mem0_chroma_path .memdb/mem0_chroma_eval \
  --llama_chroma_path .memdb/llamaindex_chroma_eval \
  --results_file benchmark/results/results.json \
  --verbose
```

---

### Explanation of Key Flags

* `--data` → dataset file to evaluate on (default: `datasets/conversations.jsonl`).
* `--compare` → which memory systems to compare (comma-separated: `keyword`, `vector`, `graph`, `mem0`, `llama`).
* `--top_k` → how many results to retrieve for evaluation (default: 5).
* `--index_backend` → vector DB backend (`chroma` or `lancedb`).
* `--collection_or_table` → collection/table name for DB storage.
* `--vector_db_path`, `--mem0_chroma_path`, `--llama_chroma_path` → local DB storage paths.
* `--restrict_to_conv` → restrict evaluation to facts within the same conversation.
* `--verbose` → print more logs while running.
* `--print_k` → how many retrieved items to show when printing examples.
* `--results_file` → path to save evaluation results (default: `results.json`).

---

### Minimal Run (with defaults)

If you just want to evaluate keyword, vector, and graph memory on the generated dataset:

```bash
python benchmark/evaluate.py --data benchmark/datasets/conversations.jsonl
```

This will save results to `results.json` in the project root.

### 3. View Metrics

Evaluation metrics can be inspected in `benchmark/metrics.py` or by opening the results file:

```bash
cat benchmark/results/results.json
```

## Running the API with Docker Compose

To build and start the API service:

```bash
docker compose up --build
```

The API will be available at [http://localhost:8888](http://localhost:8888) once the service is running.

## Project Structure

```
MemoryBridge
├── api
│   └── memory_api.py
├── benchmark
│   ├── datasets
│   │   ├── conversations.jsonl
│   │   └── promptlog.jsonl
│   ├── results
│   │   └── results.json
│   ├── evaluate.py
│   ├── generate_dataset.py
│   └── metrics.py
├── demo
│   ├── demo_graph_memory.py
│   ├── demo_keyword_baseline.py
│   ├── demo_llamaindex_memory.py
│   ├── demo_mem0_memory.py
│   └── demo_vector_memory.py
├── memory
│   ├── third_party
│   │   ├── llamaindex_adapter.py
│   │   └── mem0_adapter.py
│   ├── base.py
│   ├── graph_memory.py
│   ├── keyword_baseline.py
│   └── vector_memory.py
├── tests
│   └── test_api.py
└── utils
│   ├── config.py
│   ├── custom_embedder.py
│   ├── models.py
│   ├── reranker.py
│   └── vector_adapters.py
├── .env.example
├── docker-compose.yml
├── Dockerfile
├── README.md
└── requirements.txt
```
