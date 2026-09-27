"""Anonymous technical metrics for later prototype evaluation.

Records timing and counts only. Questions, answers, excerpts, and document
text are deliberately omitted.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from api.schemas import AskResponse


def append_interaction(path: str | Path, response: AskResponse) -> str:
    """Append one JSONL record. Returns the anonymous interaction id."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    interaction_id = str(uuid.uuid4())
    diagnostics = response.diagnostics
    record = {
        "interaction_id": interaction_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_duration_ms": diagnostics.total_duration_ms,
        "retrieval_duration_ms": diagnostics.retrieval_duration_ms,
        "rerank_duration_ms": diagnostics.rerank_duration_ms,
        "generation_duration_ms": diagnostics.generation_duration_ms,
        "retrieval_candidate_count": diagnostics.retrieval_candidate_count,
        "evidence_chunk_count": diagnostics.evidence_chunk_count,
        "insufficient_evidence": diagnostics.insufficient_evidence,
        "source_count": len(response.sources),
    }
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return interaction_id
