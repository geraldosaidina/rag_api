"""Structural checks for the technical gold dataset.

This script does not retrieve passages, rewrite queries, or generate answers.
Page bounds come from the catalogue page count, which is the PDF page count
recorded when each document was ingested.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GOLD = Path(__file__).resolve().parent / "gold_questions.json"
DEFAULT_DB = ROOT / "data" / "pfc.db"

ALLOWED_CATEGORIES = {
    "direct_fact",
    "semantic_paraphrase",
    "topic_discovery",
    "technology_methodology",
    "metadata_constrained",
    "unsupported",
    "cross_document",
}

SEMANTIC_CATEGORY = "semantic_paraphrase"


def _fail(errors: list[str], message: str) -> None:
    errors.append(message)


def _load_catalogue(db_path: Path) -> dict[str, sqlite3.Row]:
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            "SELECT id, status, page_count FROM pfc_documents"
        ).fetchall()
    finally:
        connection.close()
    return {row["id"]: row for row in rows}


def _check_semantic_test(question_id: str, semantic_test: object, errors: list[str]) -> None:
    if not isinstance(semantic_test, dict):
        _fail(errors, f"{question_id}: semantic_test must be an object")
        return
    for key in ("document_wording", "query_paraphrase", "notes"):
        value = semantic_test.get(key)
        if not isinstance(value, str) or not value.strip():
            _fail(errors, f"{question_id}: semantic_test.{key} must be non-empty")
    if not isinstance(semantic_test.get("exact_keyword_overlap_reduced"), bool):
        _fail(
            errors,
            f"{question_id}: semantic_test.exact_keyword_overlap_reduced must be a boolean",
        )


def _check_pages(
    question_id: str,
    pages: object,
    page_count: int | None,
    errors: list[str],
) -> None:
    if not isinstance(pages, list) or not pages:
        _fail(errors, f"{question_id}: evidence pdf_pages must be a non-empty list")
        return
    if page_count is None or page_count < 1:
        _fail(errors, f"{question_id}: catalogue page_count is missing")
        return
    for page in pages:
        if not isinstance(page, int) or isinstance(page, bool) or page < 1:
            _fail(errors, f"{question_id}: invalid PDF page {page!r}")
        elif page > page_count:
            _fail(
                errors,
                f"{question_id}: page {page} is outside catalogue page count {page_count}",
            )


def validate(gold_path: Path, db_path: Path) -> list[str]:
    errors: list[str] = []
    if not gold_path.is_file():
        return [f"gold dataset not found: {gold_path}"]
    if not db_path.is_file():
        return [f"catalogue not found: {db_path}"]

    dataset = json.loads(gold_path.read_text(encoding="utf-8"))
    questions = dataset.get("questions")
    if not isinstance(questions, list) or not questions:
        return ["questions must be a non-empty list"]

    catalogue = _load_catalogue(db_path)
    seen_ids: set[str] = set()

    for question in questions:
        question_id = str(question.get("id") or "")
        if not question_id:
            _fail(errors, "question id is empty")
            continue
        if question_id in seen_ids:
            _fail(errors, f"{question_id}: duplicate question id")
        seen_ids.add(question_id)

        category = question.get("category")
        if category not in ALLOWED_CATEGORIES:
            _fail(errors, f"{question_id}: category {category!r} is not allowed")

        text = question.get("question")
        if not isinstance(text, str) or not text.strip():
            _fail(errors, f"{question_id}: question text is empty")

        expected = question.get("expected")
        if not isinstance(expected, dict):
            _fail(errors, f"{question_id}: expected must be an object")
            continue

        answer_notes = expected.get("answer_notes")
        rationale = question.get("rationale")
        if not isinstance(answer_notes, str) or not answer_notes.strip():
            _fail(errors, f"{question_id}: answer_notes is empty")
        if not isinstance(rationale, str) or not rationale.strip():
            _fail(errors, f"{question_id}: rationale is empty")

        unsupported = expected.get("unsupported")
        if not isinstance(unsupported, bool):
            _fail(errors, f"{question_id}: unsupported must be a boolean")
            unsupported = False

        primary = expected.get("primary_pfc_ids")
        secondary = expected.get("secondary_pfc_ids")
        evidence = question.get("evidence")
        if not isinstance(primary, list) or not isinstance(secondary, list):
            _fail(errors, f"{question_id}: primary_pfc_ids and secondary_pfc_ids must be lists")
            continue
        if not isinstance(evidence, list):
            _fail(errors, f"{question_id}: evidence must be a list")
            continue

        if len(primary) != len(set(primary)):
            _fail(errors, f"{question_id}: duplicate id in primary_pfc_ids")
        if len(secondary) != len(set(secondary)):
            _fail(errors, f"{question_id}: duplicate id in secondary_pfc_ids")
        overlap = set(primary) & set(secondary)
        if overlap:
            _fail(errors, f"{question_id}: primary/secondary overlap {sorted(overlap)}")

        if unsupported:
            if primary or secondary or evidence:
                _fail(errors, f"{question_id}: unsupported question must have no PFC ids or evidence")
        elif not primary:
            _fail(errors, f"{question_id}: supported question needs at least one primary PFC")
        elif not evidence:
            _fail(errors, f"{question_id}: supported question needs evidence")

        referenced = set(primary) | set(secondary)
        evidence_ids: set[str] = set()
        for item in evidence:
            if not isinstance(item, dict):
                _fail(errors, f"{question_id}: evidence item must be an object")
                continue
            pfc_id = item.get("pfc_id")
            if not isinstance(pfc_id, str) or not pfc_id:
                _fail(errors, f"{question_id}: evidence pfc_id is empty")
                continue
            evidence_ids.add(pfc_id)
            if pfc_id not in referenced:
                _fail(errors, f"{question_id}: evidence {pfc_id} is not primary or secondary")
            record = catalogue.get(pfc_id)
            if record is None:
                _fail(errors, f"{question_id}: pfc_id {pfc_id} is not in the catalogue")
                continue
            if record["status"] != "indexed":
                _fail(errors, f"{question_id}: {pfc_id} is not indexed ({record['status']})")
            notes = item.get("evidence_notes")
            if not isinstance(notes, str) or not notes.strip():
                _fail(errors, f"{question_id}: evidence_notes is empty for {pfc_id}")
            _check_pages(question_id, item.get("pdf_pages"), record["page_count"], errors)

        for pfc_id in referenced:
            record = catalogue.get(pfc_id)
            if record is None:
                _fail(errors, f"{question_id}: pfc_id {pfc_id} is not in the catalogue")
            elif record["status"] != "indexed":
                _fail(errors, f"{question_id}: {pfc_id} is not indexed ({record['status']})")
            elif pfc_id not in evidence_ids and not unsupported:
                _fail(errors, f"{question_id}: {pfc_id} has no evidence entry")

        semantic_test = question.get("semantic_test", None)
        if category == SEMANTIC_CATEGORY:
            _check_semantic_test(question_id, semantic_test, errors)
        elif semantic_test is not None:
            _fail(errors, f"{question_id}: semantic_test must be null outside semantic_paraphrase")

    return errors


def main() -> int:
    gold_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_GOLD
    db_path = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_DB
    errors = validate(gold_path, db_path)
    if errors:
        print(f"INVALID: {len(errors)} error(s)")
        for error in errors:
            print(f"- {error}")
        return 1
    dataset = json.loads(gold_path.read_text(encoding="utf-8"))
    print(f"OK: {len(dataset['questions'])} questions passed structural validation")
    return 0


if __name__ == "__main__":
    sys.exit(main())
