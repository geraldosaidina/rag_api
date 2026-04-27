import logging
import sys
from pathlib import Path


# Ensure imports work when running from repository root:
# uv run python src/scripts/ingest.py
PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from rag.ingest.pipeline import IngestionException, IngestionPipeline


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    print(
        "If you changed embedding_model_name or chunking settings "
        "(chunk_size/chunk_overlap), delete data/chroma before running ingestion "
        "to rebuild a compatible index."
    )

    try:
        pipeline = IngestionPipeline()
        result = pipeline.run()
        print(result)
        return 0
    except IngestionException as exc:
        print(f"Ingestion failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
