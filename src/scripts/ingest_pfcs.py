"""Ingest the real PFC drop folder into the catalogue and pfc_corpus.

Run from the repository root:
  uv run python src/scripts/ingest_pfcs.py --rebuild
  uv run python src/scripts/ingest_pfcs.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from rag.documents.models import PfcStatus
from rag.service.factory import build_pfc_ingestion_service, load_rag_app_config


def _delete_production_collection(persist_directory: str, collection_name: str) -> None:
    import chromadb

    client = chromadb.PersistentClient(path=persist_directory)
    names = [collection.name for collection in client.list_collections()]
    if collection_name not in names:
        print(f"Collection '{collection_name}' was not present.")
        return
    client.delete_collection(collection_name)
    print(f"Deleted Chroma collection '{collection_name}'.")


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    parser = argparse.ArgumentParser(description="Ingest PFC PDFs into the catalogue.")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete the target Chroma collection before ingesting.",
    )
    parser.add_argument(
        "--directory",
        default=None,
        help="PDF drop directory. Defaults to PFC_SAMPLES_DIRECTORY.",
    )
    args = parser.parse_args()
    config = load_rag_app_config()
    if args.rebuild:
        _delete_production_collection(config.persist_directory, config.collection_name)

    service = build_pfc_ingestion_service(config)
    directory = args.directory or config.pfc_samples_directory
    records = service.ingest_directory(directory)
    counts = {status.value: 0 for status in PfcStatus}
    for record in records:
        counts[record.status.value] = counts.get(record.status.value, 0) + 1
        if record.status == PfcStatus.FAILED:
            print(f"FAILED {record.original_filename}: {record.error}")
    print(
        f"Ingested {len(records)} PDF(s) from {directory}. "
        f"indexed={counts['indexed']} failed={counts['failed']} "
        f"processing={counts['processing']} registered={counts['registered']} "
        f"chunks={service.vector_store.count()} collection={config.collection_name}"
    )
    return 0 if counts["processing"] == 0 and counts["registered"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
