"""Re-evaluate Phase 2 LLM judgments without regenerating answers.

Reads an existing run directory, recovers unambiguous markdown scores from
failed JSON parses, optionally runs a second structured judge for remaining
cases, and writes artefacts under <run>/reevaluation/.

Does not modify original generation.jsonl / judgments.jsonl / summary.json.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evaluation.judging.llm_judge import (  # noqa: E402
    JUDGE_SCALE,
    SYSTEM_PROMPT,
    build_user_prompt,
    judge_prompt_record,
)
from rag.llm.ollama_client import LLMConfig, LLMException, OllamaLLMClient  # noqa: E402
from rag.service.factory import load_rag_app_config  # noqa: E402

SCORE_KEYS = (
    "faithfulness",
    "answer_relevance",
    "citation_semantic_support",
    "unsupported_handling",
)
DIM_PATTERNS = {
    "faithfulness": re.compile(r"\*\*Faithfulness\*\*\s*:\s*([^\n]+)", re.I),
    "answer_relevance": re.compile(r"\*\*Answer Relevance\*\*\s*:\s*([^\n]+)", re.I),
    "citation_semantic_support": re.compile(
        r"\*\*Citation Semantic Support\*\*\s*:\s*([^\n]+)", re.I
    ),
    "unsupported_handling": re.compile(
        r"\*\*Unsupported Handling\*\*\s*:\s*([^\n]+)", re.I
    ),
}
SECOND_SYSTEM = (
    SYSTEM_PROMPT
    + "\n8. Output MUST be a single JSON object. No markdown, no prose before/after.\n"
    "9. Scores MUST be integers 0, 1, or 2 only (never 1.5).\n"
    "10. unsupported_handling must be null when the question is not unsupported.\n"
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _parse_score_token(token: str) -> tuple[str, int | None]:
    text = token.strip()
    if re.match(r"^(null|none|n/?a|nenhum[a]?)\b", text, re.I):
        return "null", None
    match = re.match(r"^(\d+(?:\.\d+)?)\s*(?:/\s*2)?\b", text)
    if not match:
        return "unparseable", None
    number = float(match.group(1))
    if number != int(number):
        return "fractional", None
    value = int(number)
    if value not in (0, 1, 2):
        return "out_of_range", None
    return "ok", value


def analyze_parse_errors(judges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for judge in judges:
        if not judge.get("parse_error"):
            continue
        raw = judge.get("raw_response") or ""
        cause = {
            "question_id": judge["question_id"],
            "recorded_parse_error": judge["parse_error"],
            "raw_response_length": len(raw),
            "contains_json_object_braces": "{" in raw and "}" in raw,
            "contains_markdown_fences": "```" in raw,
            "looks_like_markdown_rubric": bool(
                re.search(r"\*\*Faithfulness\*\*", raw, re.I)
            ),
            "root_cause": (
                "Judge returned Portuguese/markdown prose with **Dimension**: score "
                "labels instead of a JSON object. json.loads fails with "
                "'Expecting value: line 1 column 1 (char 0)' because the first "
                "character is not '{' or '['."
            ),
            "dimension_tokens": {},
        }
        for dim, pattern in DIM_PATTERNS.items():
            match = pattern.search(raw)
            if not match:
                cause["dimension_tokens"][dim] = {
                    "present": False,
                    "status": "missing",
                    "token": None,
                }
                continue
            status, value = _parse_score_token(match.group(1))
            cause["dimension_tokens"][dim] = {
                "present": True,
                "status": status,
                "token": match.group(1).strip()[:80],
                "integer_if_ok": value,
            }
        rows.append(cause)
    return rows


def try_recover_from_raw(
    judge: dict[str, Any],
    *,
    unsupported: bool,
) -> dict[str, Any]:
    raw = judge.get("raw_response") or ""
    scores: dict[str, Any] = {}
    recovery_detail: dict[str, Any] = {}
    recoverable = True
    for dim, pattern in DIM_PATTERNS.items():
        match = pattern.search(raw)
        if not match:
            recovery_detail[dim] = {"status": "missing"}
            if dim != "unsupported_handling" or unsupported:
                recoverable = False
            continue
        status, value = _parse_score_token(match.group(1))
        recovery_detail[dim] = {
            "status": status,
            "token": match.group(1).strip()[:80],
            "value": value,
        }
        if dim == "unsupported_handling":
            if not unsupported:
                scores[dim] = None
            elif status == "ok":
                scores[dim] = value
            else:
                recoverable = False
            continue
        if status != "ok":
            recoverable = False
        else:
            scores[dim] = value

    needs_human = True
    reasons = ["recovered_from_markdown_raw_response"]
    if recoverable:
        if any(scores.get(k) == 1 for k in SCORE_KEYS[:3]):
            reasons.append("mid_score")
        if unsupported and scores.get("unsupported_handling") in {0, 1}:
            reasons.append("unsupported_not_clearly_handled")
        if not any(
            r in reasons
            for r in ("mid_score", "unsupported_not_clearly_handled")
        ) and not unsupported:
            # Still flag recovery provenance; human review recommended for mid/auto flags only
            needs_human = any(scores.get(k) == 1 for k in SCORE_KEYS[:3])
        elif unsupported:
            needs_human = scores.get("unsupported_handling") in {0, 1, None} or any(
                scores.get(k) == 1 for k in SCORE_KEYS[:3]
            )
        else:
            needs_human = any(scores.get(k) == 1 for k in SCORE_KEYS[:3])

    return {
        "question_id": judge["question_id"],
        "source": "recovered_from_original_raw_response",
        "method": (
            "Regex extraction of **Dimension**: <token> from original raw_response; "
            "accept only integer tokens 0|1|2 (optional /2). Reject fractional "
            "(e.g. 1.5), missing, or unparseable tokens. Non-unsupported questions "
            "force unsupported_handling=null."
        ),
        "recoverable": recoverable,
        "recovery_detail": recovery_detail,
        "model_name": judge.get("model_name"),
        "system_prompt": judge.get("system_prompt"),
        "scale": judge.get("scale"),
        "faithfulness": scores.get("faithfulness") if recoverable else None,
        "answer_relevance": scores.get("answer_relevance") if recoverable else None,
        "citation_semantic_support": (
            scores.get("citation_semantic_support") if recoverable else None
        ),
        "unsupported_handling": (
            scores.get("unsupported_handling") if recoverable else None
        ),
        "needs_human_review": True if not recoverable else needs_human,
        "human_review_reasons": (
            ["not_unequivocally_recoverable"] + [
                f"{dim}:{recovery_detail[dim]['status']}"
                for dim in SCORE_KEYS
                if recovery_detail.get(dim, {}).get("status")
                not in {"ok", "null"}
                and not (
                    dim == "unsupported_handling"
                    and not unsupported
                    and recovery_detail.get(dim, {}).get("status")
                    in {"null", "unparseable", "missing", "ok"}
                )
            ]
            if not recoverable
            else reasons
        ),
        "original_parse_error": judge.get("parse_error"),
        "original_raw_response": raw,
    }


def _validate_judge_schema(
    parsed: dict[str, Any],
    *,
    unsupported: bool,
) -> tuple[dict[str, Any] | None, str | None]:
    required = [
        "faithfulness",
        "answer_relevance",
        "citation_semantic_support",
        "unsupported_handling",
        "rationale",
        "needs_human_review",
        "human_review_reasons",
    ]
    missing = [key for key in required if key not in parsed]
    if missing:
        return None, f"missing_keys:{missing}"

    scores: dict[str, Any] = {}
    for key in SCORE_KEYS[:3]:
        try:
            value = int(parsed[key])
        except (TypeError, ValueError):
            return None, f"non_integer:{key}"
        if value not in (0, 1, 2):
            return None, f"out_of_range:{key}={value}"
        scores[key] = value

    raw_u = parsed.get("unsupported_handling")
    if unsupported:
        try:
            value = int(raw_u)
        except (TypeError, ValueError):
            return None, "non_integer:unsupported_handling"
        if value not in (0, 1, 2):
            return None, f"out_of_range:unsupported_handling={value}"
        scores["unsupported_handling"] = value
    else:
        if raw_u is not None:
            return None, "unsupported_handling_must_be_null"
        scores["unsupported_handling"] = None

    reasons = parsed.get("human_review_reasons")
    if not isinstance(reasons, list):
        return None, "human_review_reasons_not_list"

    return {
        **scores,
        "rationale": str(parsed.get("rationale") or "").strip(),
        "needs_human_review": bool(parsed.get("needs_human_review")),
        "human_review_reasons": [str(item) for item in reasons],
    }, None


def _extract_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("not_a_json_object")
    return parsed


def second_judge(
    client: OllamaLLMClient,
    *,
    generation_row: dict[str, Any],
    answer_notes: str | None,
    max_attempts: int = 2,
) -> dict[str, Any]:
    unsupported = bool(generation_row["unsupported"])
    user_prompt = build_user_prompt(
        question=generation_row["question"],
        answer=generation_row["answer"],
        evidence_context_text=generation_row.get("evidence_context_text") or "",
        sources=generation_row.get("sources") or [],
        unsupported=unsupported,
        answer_notes=answer_notes,
    )
    attempts: list[dict[str, Any]] = []
    for attempt in range(1, max_attempts + 1):
        try:
            response = client.invoke_messages(
                [("system", SECOND_SYSTEM), ("user", user_prompt)]
            )
            raw = response.content
            model_name = response.model_name
        except LLMException as exc:
            attempts.append({"attempt": attempt, "error": str(exc), "raw": ""})
            continue
        try:
            parsed = _extract_json_object(raw)
            validated, schema_error = _validate_judge_schema(
                parsed, unsupported=unsupported
            )
            attempts.append(
                {
                    "attempt": attempt,
                    "raw": raw,
                    "schema_error": schema_error,
                    "parsed_keys": sorted(parsed.keys()),
                }
            )
            if validated is None:
                continue
            needs_human = validated["needs_human_review"]
            reasons = list(validated["human_review_reasons"])
            reasons.append("second_evaluation")
            if any(validated[k] == 1 for k in SCORE_KEYS[:3]):
                needs_human = True
                if "mid_score" not in reasons:
                    reasons.append("mid_score")
            if unsupported and validated["unsupported_handling"] in {0, 1}:
                needs_human = True
                if "unsupported_not_clearly_handled" not in reasons:
                    reasons.append("unsupported_not_clearly_handled")
            return {
                "question_id": generation_row["question_id"],
                "source": "second_evaluation",
                "evaluation_label": "second_automatic_evaluation",
                "note": (
                    "This is a new judgment of the same frozen answer/evidence; "
                    "it does not replace or rewrite the original failed judgment."
                ),
                "model_name": model_name,
                "system_prompt": SECOND_SYSTEM,
                "scale": JUDGE_SCALE,
                "schema_validated": True,
                "attempts": attempts,
                "faithfulness": validated["faithfulness"],
                "answer_relevance": validated["answer_relevance"],
                "citation_semantic_support": validated["citation_semantic_support"],
                "unsupported_handling": validated["unsupported_handling"],
                "rationale": validated["rationale"],
                "needs_human_review": needs_human,
                "human_review_reasons": reasons,
            }
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            attempts.append(
                {"attempt": attempt, "raw": raw, "parse_error": str(exc)}
            )
    return {
        "question_id": generation_row["question_id"],
        "source": "second_evaluation",
        "evaluation_label": "second_automatic_evaluation",
        "note": (
            "Second evaluation attempted but produced no schema-valid scores; "
            "scores left null (not converted to zero)."
        ),
        "model_name": client.config.model_name,
        "system_prompt": SECOND_SYSTEM,
        "scale": JUDGE_SCALE,
        "schema_validated": False,
        "attempts": attempts,
        "faithfulness": None,
        "answer_relevance": None,
        "citation_semantic_support": None,
        "unsupported_handling": None,
        "rationale": "",
        "needs_human_review": True,
        "human_review_reasons": ["second_evaluation_schema_failed"],
    }


def classify_citation_errors(
    generation_rows: list[dict[str, Any]],
    gold_pages: dict[str, dict[str, list[int]]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in generation_rows:
        qid = row["question_id"]
        primary = {str(x) for x in row.get("primary_relevant") or []}
        unsupported = bool(row["unsupported"])
        metrics = row.get("citation_metrics") or {}
        pages_by_pfc = gold_pages.get(qid, {})
        wrong_docs: list[dict[str, Any]] = []
        page_mismatches: list[dict[str, Any]] = []
        for source in row.get("sources") or []:
            pfc_id = str(source.get("pfc_id") or "")
            page = source.get("page")
            if not unsupported and primary and pfc_id and pfc_id not in primary:
                wrong_docs.append(
                    {
                        "citation_id": source.get("citation_id"),
                        "pfc_id": pfc_id,
                        "page": page,
                        "title": source.get("title"),
                    }
                )
            gold_pages_for = pages_by_pfc.get(pfc_id)
            if (
                gold_pages_for is not None
                and page is not None
                and int(page) not in {int(p) for p in gold_pages_for}
            ):
                page_mismatches.append(
                    {
                        "citation_id": source.get("citation_id"),
                        "pfc_id": pfc_id,
                        "cited_page": page,
                        "gold_pages": list(gold_pages_for),
                    }
                )
        out.append(
            {
                "question_id": qid,
                "unsupported": unsupported,
                "wrong_document_citations": wrong_docs,
                "has_wrong_document": bool(wrong_docs),
                "page_mismatches_vs_gold": page_mismatches,
                "has_page_mismatch": bool(page_mismatches),
                "structurally_invalid_citations_removed": metrics.get(
                    "invalid_citations_removed"
                )
                or [],
                "has_structural_invalid": int(metrics.get("invalid_citation_count") or 0)
                > 0,
                "citation_doc_precision": metrics.get("citation_doc_precision"),
                "citation_doc_recall": metrics.get("citation_doc_recall"),
                "citation_page_precision": metrics.get("citation_page_precision"),
                "note_on_unsupported_claim": (
                    "Whether cited passages support answer claims is a semantic "
                    "judgment (LLM/human), not derived solely from gold doc/page hits."
                ),
            }
        )
    return out


def build_coverage(
    *,
    original_ok: list[str],
    recovered: list[dict[str, Any]],
    second: list[dict[str, Any]],
    all_ids: list[str],
    generation_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    recovered_ok = [r for r in recovered if r["recoverable"]]
    recovered_fail = [r for r in recovered if not r["recoverable"]]
    second_ok = [r for r in second if r.get("schema_validated")]
    second_fail = [r for r in second if not r.get("schema_validated")]

    valid_scores: dict[str, dict[str, Any]] = {}
    for qid in original_ok:
        # filled later by caller with original judge payload
        valid_scores[qid] = {"provenance": "original_json_judgment"}
    for row in recovered_ok:
        valid_scores[row["question_id"]] = {
            "provenance": "recovered_from_original_raw_response",
            "faithfulness": row["faithfulness"],
            "answer_relevance": row["answer_relevance"],
            "citation_semantic_support": row["citation_semantic_support"],
            "unsupported_handling": row["unsupported_handling"],
            "needs_human_review": row["needs_human_review"],
        }
    for row in second_ok:
        valid_scores[row["question_id"]] = {
            "provenance": "second_evaluation",
            "faithfulness": row["faithfulness"],
            "answer_relevance": row["answer_relevance"],
            "citation_semantic_support": row["citation_semantic_support"],
            "unsupported_handling": row["unsupported_handling"],
            "needs_human_review": row["needs_human_review"],
        }

    pending = [qid for qid in all_ids if qid not in valid_scores]
    needs_human = []
    for qid, payload in valid_scores.items():
        gen = generation_by_id[qid]
        flag = bool(payload.get("needs_human_review"))
        cm = gen.get("citation_metrics") or {}
        if (
            not gen["unsupported"]
            and cm.get("citation_doc_precision") not in (None, 1.0)
        ):
            flag = True
        if int(cm.get("invalid_citation_count") or 0) > 0:
            flag = True
        if flag:
            needs_human.append(qid)

    return {
        "total_questions": len(all_ids),
        "valid_automatic_scores_count": len(valid_scores),
        "valid_by_provenance": {
            "original_json_judgment": len(original_ok),
            "recovered_from_original_raw_response": len(recovered_ok),
            "second_evaluation": len(second_ok),
        },
        "pending_without_valid_scores": pending,
        "pending_count": len(pending),
        "recovery_attempted_not_unequivocal": [
            r["question_id"] for r in recovered_fail
        ],
        "second_evaluation_schema_failed": [r["question_id"] for r in second_fail],
        "needs_human_review_question_ids": sorted(needs_human),
        "needs_human_review_count": len(needs_human),
        "human_validation_performed": False,
        "means_policy": (
            "Do not compute means over questions lacking valid scores for that "
            "dimension. Means, if computed later, must use only valid_scores."
        ),
        "valid_scores": valid_scores,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=ROOT
        / "evaluation"
        / "results"
        / "runs"
        / "20261019T210644Z-d128e825",
    )
    parser.add_argument("--skip-second-judge", action="store_true")
    args = parser.parse_args()
    run_dir = args.run_dir
    out_dir = run_dir / "reevaluation"
    out_dir.mkdir(parents=True, exist_ok=True)

    judges = _load_jsonl(run_dir / "judgments.jsonl")
    generations = _load_jsonl(run_dir / "generation.jsonl")
    gen_by_id = {row["question_id"]: row for row in generations}
    gold = json.loads(
        (ROOT / "evaluation" / "gold_questions.json").read_text(encoding="utf-8")
    )
    notes = {
        str(item["id"]): str(item["expected"].get("answer_notes") or "")
        for item in gold["questions"]
    }
    gold_pages: dict[str, dict[str, list[int]]] = {}
    for item in gold["questions"]:
        by_pfc: dict[str, list[int]] = {}
        for evidence in item.get("evidence") or []:
            pfc_id = str(evidence["pfc_id"])
            by_pfc.setdefault(pfc_id, [])
            for page in evidence.get("pdf_pages") or []:
                by_pfc[pfc_id].append(int(page))
        gold_pages[str(item["id"])] = by_pfc

    parse_analysis = analyze_parse_errors(judges)
    (out_dir / "parse_error_analysis.json").write_text(
        json.dumps(parse_analysis, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    original_ok_ids = [
        j["question_id"] for j in judges if not j.get("parse_error")
    ]
    err_judges = [j for j in judges if j.get("parse_error")]
    recovered: list[dict[str, Any]] = []
    for judge in err_judges:
        gen = gen_by_id[judge["question_id"]]
        recovered.append(
            try_recover_from_raw(judge, unsupported=bool(gen["unsupported"]))
        )
    _write_jsonl(out_dir / "recovered_judgments.jsonl", recovered)

    need_second = [r["question_id"] for r in recovered if not r["recoverable"]]
    second_rows: list[dict[str, Any]] = []
    if need_second and not args.skip_second_judge:
        app_config = load_rag_app_config()
        client = OllamaLLMClient(
            LLMConfig(
                model_name=app_config.llm_model_name,
                base_url=app_config.ollama_base_url,
                temperature=0.0,
            )
        )
        if not client.health_check():
            raise RuntimeError(
                "Ollama unreachable for second evaluation. "
                "Re-run later or pass --skip-second-judge."
            )
        print(f"Second evaluation for: {need_second}", flush=True)
        for qid in need_second:
            print(f"  judging {qid}...", flush=True)
            second_rows.append(
                second_judge(
                    client,
                    generation_row=gen_by_id[qid],
                    answer_notes=notes.get(qid) or None,
                )
            )
            print(
                f"    validated={second_rows[-1].get('schema_validated')} "
                f"F={second_rows[-1].get('faithfulness')} "
                f"R={second_rows[-1].get('answer_relevance')} "
                f"C={second_rows[-1].get('citation_semantic_support')} "
                f"U={second_rows[-1].get('unsupported_handling')}",
                flush=True,
            )
    elif need_second:
        print("Skipping second judge; pending:", need_second, flush=True)
    _write_jsonl(out_dir / "second_judgments.jsonl", second_rows)

    coverage = build_coverage(
        original_ok=original_ok_ids,
        recovered=recovered,
        second=second_rows,
        all_ids=[row["question_id"] for row in generations],
        generation_by_id=gen_by_id,
    )
    for judge in judges:
        if judge.get("parse_error"):
            continue
        coverage["valid_scores"][judge["question_id"]] = {
            "provenance": "original_json_judgment",
            "faithfulness": judge.get("faithfulness"),
            "answer_relevance": judge.get("answer_relevance"),
            "citation_semantic_support": judge.get("citation_semantic_support"),
            "unsupported_handling": judge.get("unsupported_handling"),
            "needs_human_review": judge.get("needs_human_review"),
        }
    needs_human_ids: list[str] = []
    for qid, payload in coverage["valid_scores"].items():
        gen = gen_by_id[qid]
        flag = bool(payload.get("needs_human_review"))
        cm = gen.get("citation_metrics") or {}
        if not gen["unsupported"] and cm.get("citation_doc_precision") not in (
            None,
            1.0,
        ):
            flag = True
        if int(cm.get("invalid_citation_count") or 0) > 0:
            flag = True
        if any(payload.get(k) == 1 for k in SCORE_KEYS[:3]):
            flag = True
        if gen["unsupported"] and payload.get("unsupported_handling") in {0, 1, None}:
            flag = True
        if flag:
            needs_human_ids.append(qid)
    coverage["needs_human_review_question_ids"] = sorted(set(needs_human_ids))
    coverage["needs_human_review_count"] = len(
        coverage["needs_human_review_question_ids"]
    )
    coverage["reevaluation_id"] = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-"
        + uuid.uuid4().hex[:8]
    )
    coverage["source_run_id"] = run_dir.name
    coverage["created_utc"] = datetime.now(timezone.utc).isoformat()
    (out_dir / "judgment_coverage.json").write_text(
        json.dumps(coverage, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    citation_errors = classify_citation_errors(generations, gold_pages)
    # attach semantic flags from valid scores only
    for item in citation_errors:
        scores = coverage["valid_scores"].get(item["question_id"])
        if scores and scores.get("citation_semantic_support") is not None:
            item["semantic_support_score_if_valid"] = scores[
                "citation_semantic_support"
            ]
            item["possible_unsupported_claim_by_cited_passage"] = (
                scores["citation_semantic_support"] == 0
            )
        else:
            item["semantic_support_score_if_valid"] = None
            item["possible_unsupported_claim_by_cited_passage"] = None
    (out_dir / "citation_error_analysis.json").write_text(
        json.dumps(citation_errors, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    (out_dir / "second_judge_prompt_record.json").write_text(
        json.dumps(
            {
                **judge_prompt_record(load_rag_app_config().llm_model_name),
                "system_prompt": SECOND_SYSTEM,
                "schema_validation": {
                    "required_keys": list(SCORE_KEYS)
                    + ["rationale", "needs_human_review", "human_review_reasons"],
                    "score_domain": [0, 1, 2],
                    "unsupported_handling_null_if_supported": True,
                    "fractional_scores_rejected": True,
                    "failed_schema_scores_left_null": True,
                },
                "label": "second_automatic_evaluation",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(json.dumps({
        "out_dir": str(out_dir),
        "parse_errors": len(parse_analysis),
        "recovered": sum(1 for r in recovered if r["recoverable"]),
        "need_second": need_second,
        "second_ok": sum(1 for r in second_rows if r.get("schema_validated")),
        "valid": coverage["valid_automatic_scores_count"],
        "pending": coverage["pending_count"],
        "needs_human": coverage["needs_human_review_count"],
    }, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
