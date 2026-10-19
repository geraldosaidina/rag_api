# Phase 2 judging protocol (H1 / H4)

## Purpose

Evaluate grounded generation and documentary references for the production RAG stack:

- retrieval: hybrid
- rewrite: rules + LLM
- reranker: on
- generation: configured Ollama model

Gold `answer_notes` are **factual expectations for judges**, not canonical answer strings. Do not score exact lexical match against `answer_notes`.

## Evidence available for judging

For each question the runner stores:

- `question`, `answer` (after citation cleanup)
- `raw_answer_before_citation_cleanup`
- `evidence_context_text` and `evidence_items[]` (exact fragments given to the model)
- `sources[]` with `citation_id`, `pfc_id`, `page`, excerpt, title, year
- `diagnostics.insufficient_evidence`, timings, `invalid_citations_removed`
- gold `primary_pfc_ids`, `unsupported`, and `evidence.pdf_pages`

## Dimensions

### 1. Faithfulness (to supplied evidence)

Does every factual claim in the answer follow from `evidence_context_text` / `evidence_items`?

| Score | Meaning |
|------:|---------|
| 2 | All material claims are supported by the supplied evidence |
| 1 | Mostly supported; minor unsupported detail or over-generalisation |
| 0 | Material unsupported claim, contradiction of evidence, or invented fact |

Uses **only** the evidence pack given to the model, not the full PDF corpus.

### 2. Answer relevance

Does the answer address the user question?

| Score | Meaning |
|------:|---------|
| 2 | Directly answers the question (or correctly abstains when unsupported) |
| 1 | Partially on-topic |
| 0 | Off-topic or ignores the question |

### 3. Citation structural validity

Automatic:

- labels present in the answer must exist in the evidence pack (`CitationValidator`)
- `invalid_citations_removed` counted

### 4. Citation semantic support

For each cited label used after a claim: does the **cited fragment** support that claim?

| Score | Meaning |
|------:|---------|
| 2 | Cited passages support the adjacent claims |
| 1 | Mixed / weak support |
| 0 | Citation window does not support the claim, or decorative citations |

Distinct from document/page hit rates against gold.

### 5. Citation gold alignment (automatic)

- Document precision/recall vs `primary_pfc_ids`
- Page precision vs `evidence.pdf_pages` when the cited `pfc_id` has gold pages

### 6. Unsupported questions (Q023–Q026)

Evaluate separately. Success means the system does **not** present unfounded information as established fact matching the false/absent premise.

Automatic signals (not sufficient alone):

- `insufficient_evidence == true`
- abstention phrases in Portuguese
- citation gold alignment is undefined (empty relevant set)

LLM/human judge:

| Score | Meaning |
|------:|---------|
| 2 | Correctly refuses or states evidence does not establish the request |
| 1 | Hedged but still suggests a matching document without clear correction |
| 0 | Affirms false premise or invents matching PFCs/facts |

## Human review required when

- LLM judge score is 1 on any dimension
- Unsupported question score is 0 or 1
- `citation_doc_precision < 1` on a supported question that cites sources
- `invalid_citation_count > 0`
- Judge JSON parse failure or model error
- Answer claims catalogue metadata that conflicts with evidence metadata blocks

## What not to do

- Do not treat `answer_notes` as a gold string for ROUGE/exact match
- Do not mark faithfulness using the full PDF outside the evidence pack
- Do not declare H1/H4 confirmed from automatic scores alone
