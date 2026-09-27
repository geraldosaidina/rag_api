"""Smoke questions against the ingested PFC corpus.

Run after ingest_pfcs.py:
  uv run python src/scripts/test_pfc_smoke.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from rag.documents.models import PfcStatus
from rag.service.factory import build_rag_service

QUESTIONS = (
    "Que projectos finais tratam de redes sem fios ou de cobertura de rede móvel?",
    "Como é que o projecto de Edson Samuel Langa usa RFID no ensino no ISUTC?",
    "Quem escreveu o projecto sobre videovigilância IP na Avenida 25 de Setembro e em que ano?",
    "Que projectos finais da Universidade Eduardo Mondlane estão nesta colecção?",
)


def main() -> int:
    service = build_rag_service()
    catalogue = service.catalogue
    records = catalogue.list_documents() if catalogue is not None else []
    print(f"Catalogue rows: {len(records)}")
    for record in records:
        print(
            f"- {record.status.value:10} {record.year or '-':4} "
            f"{record.original_filename} | {(record.title or '')[:80]}"
        )
    survey = service.retriever.vector_store.count_where(
        {"source": "2503.10677v2.pdf"}
    )
    print(f"Survey chunks in pfc_corpus: {survey}")
    print(f"Total chunks: {service.retriever.vector_store.count()}")

    for question in QUESTIONS:
        started = time.perf_counter()
        result = service.answer(question)
        elapsed = time.perf_counter() - started
        diagnostics = result.diagnostics
        print("\n" + "=" * 80)
        print("Q:", question)
        print(f"elapsed_s={elapsed:.1f} total_ms={diagnostics.total_duration_ms:.0f}")
        print(
            "retrieval_ms="
            f"{diagnostics.retrieval_duration_ms} rerank_ms={diagnostics.rerank_duration_ms} "
            f"generation_ms={diagnostics.generation_duration_ms} "
            f"candidates={diagnostics.retrieval_candidate_count} "
            f"evidence={diagnostics.evidence_chunk_count} "
            f"insufficient={diagnostics.insufficient_evidence}"
        )
        print(result.answer)
        for source in result.sources:
            authors = ", ".join(source.authors)
            print(
                f"  {source.citation_id} pfc_id={source.pfc_id} "
                f"title={source.title} authors={authors} year={source.year} "
                f"page={source.page} file={source.source}"
            )
            if "2503.10677" in (source.source or ""):
                print("  ERROR: survey source leaked into the answer")
                return 1
    failed = [record for record in records if record.status == PfcStatus.FAILED]
    print(f"\nFailed catalogue rows: {len(failed)}")
    for record in failed:
        print(f"- {record.original_filename}: {record.error}")
    return 0 if survey == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
