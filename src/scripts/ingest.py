import logging
import sys
from pathlib import Path


# Ensure imports work when running from repository root:
# uv run python src/scripts/ingest.py
PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from rag.ingest.pipeline import IngestionConfig, IngestionException, IngestionPipeline


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    print(
        "Legacy survey ingestion writes collection 'survey_corpus'. "
        "It does not modify the production pfc_corpus collection."
    )

    try:
        pipeline = IngestionPipeline(
            IngestionConfig(collection_name="survey_corpus")
        )
        result = pipeline.run()
        print(result)
        return 0
    except IngestionException as exc:
        print(f"Ingestion failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
