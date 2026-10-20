# Evaluation (Phase 1 — retrieval benchmark)

## What this folder contains

| Path | Role |
|------|------|
| `gold_questions.json` | Gold set (28 questions, document-level relevance) |
| `gold_questions.md` | Human-readable companion |
| `validate_gold_dataset.py` | Structural validation only |
| `metrics/retrieval.py` | Precision@K, Recall@K, MRR helpers |
| `run_benchmark.py` | Offline runner for configurations A1–A5 |
| `results/runs/<run_id>/` | Manifest, per-query JSONL, summary |

Phase 1 does **not** generate answers, score faithfulness, or call the student API.

## Configurations (A1–A5)

| Id | Retrieval | Rewrite |
|----|-----------|---------|
| A1 | `bm25` | `none` |
| A2 | `dense` | `none` |
| A3 | `hybrid` | `none` |
| A4 | `hybrid` | `rules` |
| A5 | `hybrid` | `rules_llm` |

Rewrite `none` sets `QueryRewriteConfig(enabled=False)` so **rules and LLM are off**.

## Run

```bash
uv run python evaluation/validate_gold_dataset.py
uv run python evaluation/run_benchmark.py
uv run python evaluation/run_benchmark.py --configs A1,A2,A3
```

Non-regression:

```bash
uv run python src/scripts/test_benchmark_phase1.py
uv run python src/scripts/test_slice1_baseline.py
```

## Relevance protocol and annotation limitations

- Relevant set = `expected.primary_pfc_ids` only.
- Rankings are collapsed to the first occurrence of each `pfc_id`.
- `secondary_pfc_ids` are empty in gold v1.0 and are ignored.
- `evidence.pdf_pages` are **not** used for Phase 1 IR metrics.
- Queries with `unsupported=true` are excluded from macro averages of Precision@5, Recall@5, Recall@10, and MRR; their rankings are still stored.
- Relevance is binary and document-level, not graded and not chunk-level.

## Phase 2 — generation and references (B1)

Production stack only: hybrid + rules_llm + rerank + generation.

```bash
uv run python evaluation/run_generation_benchmark.py
uv run python evaluation/run_generation_benchmark.py --skip-llm-judge
uv run python src/scripts/test_benchmark_phase2.py
```

Outputs under `results/runs/<run_id>/`:

| File | Content |
|------|---------|
| `generation.jsonl` | Question, answer, evidence pack, sources, timings, automatic citation metrics |
| `judgments.jsonl` | LLM-as-judge scores, prompts metadata, human-review flags |
| `summary.json` | Macro averages (supported vs unsupported separated) |
| `run_manifest.json` | Models, corpus id, judge protocol reference |

Judging protocol: `judging/protocol.md`.

`answer_notes` in gold are judge hints only, never canonical answers.

## Production chatbot

The student API and frontend are unchanged. `HybridRetriever.retrieve` defaults to `retrieval_mode="hybrid"`. `RAGService` does not pass `retrieval_mode`. Production rewrite still uses `enabled=True` with `use_llm` controlled by `RAG_USE_LLM_REWRITE`.

`AnswerResult` may carry optional evidence traces for offline evaluation; `api.mapping.to_ask_response` does not expose them to students.
