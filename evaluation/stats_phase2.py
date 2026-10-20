"""Phase 2 human-score summaries, judge agreement, and error-case tables."""

from __future__ import annotations

from typing import Any

from evaluation.stats_lib import (
    AGREEMENT_DIMENSIONS,
    EXPECTED_IDS,
    HUMAN_DIMENSIONS,
    UNSUPPORTED_IDS,
    AnalysisError,
    bootstrap_mean_ci,
    confusion_counts,
    describe,
    latency_describe,
    load_jsonl,
    quadratic_weighted_kappa,
    validate_id_set,
)


def index_by_id(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        qid = str(row["question_id"])
        if qid in by_id:
            raise AnalysisError(f"Duplicate {label} row: {qid}")
        by_id[qid] = row
    validate_id_set(by_id.keys(), label)
    return by_id


def load_generation(path) -> dict[str, dict[str, Any]]:
    return index_by_id(load_jsonl(path), "phase2 generation")


def score_table(coverage: dict[str, Any]) -> dict[str, dict[str, Any]]:
    scores = coverage.get("valid_scores") or {}
    if set(scores) - set(EXPECTED_IDS):
        raise AnalysisError(f"Unexpected automatic score ids: {sorted(set(scores) - set(EXPECTED_IDS))}")
    return scores


def _subset_ids(name: str) -> list[str]:
    if name == "supported":
        return [qid for qid in EXPECTED_IDS if qid not in UNSUPPORTED_IDS]
    if name == "unsupported":
        return list(UNSUPPORTED_IDS)
    if name == "all":
        return list(EXPECTED_IDS)
    raise AnalysisError(name)


def human_stat_rows(human: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for subset in ("supported", "unsupported", "all"):
        ids = _subset_ids(subset)
        dimensions = list(HUMAN_DIMENSIONS)
        if subset != "unsupported":
            dimensions = [
                name for name in dimensions if name != "human_unsupported_handling"
            ]
        for dimension in dimensions:
            values = [human[qid][dimension] for qid in ids]
            valid = [value for value in values if value is not None]
            stats = describe(valid)
            ci = bootstrap_mean_ci(valid, seed_key=f"human:{subset}:{dimension}")
            for score in (0, 1, 2):
                count = sum(1 for value in valid if value == score)
                rows.append(
                    {
                        "record_type": "frequency",
                        "subset": subset,
                        "dimension": dimension,
                        "score": score,
                        "count": count,
                        "percentage": (100.0 * count / len(valid)) if valid else None,
                        "n_valid": len(valid),
                        "n_expected": len(ids),
                        "n_missing": len(ids) - len(valid),
                        "mean": stats["mean"],
                        "median": stats["median"],
                        "std": stats["std"],
                        "ci95_low": ci["ci95_low"],
                        "ci95_high": ci["ci95_high"],
                    }
                )
            if dimension == "human_archive_correctness":
                n = len(valid)
                rows.append(
                    {
                        "record_type": "archive_rate",
                        "subset": subset,
                        "dimension": dimension,
                        "score": "",
                        "count": n,
                        "percentage": None,
                        "n_valid": n,
                        "n_expected": len(ids),
                        "n_missing": len(ids) - n,
                        "rate_eq_2": (sum(v == 2 for v in valid) / n) if n else None,
                        "rate_ge_1": (sum(v >= 1 for v in valid) / n) if n else None,
                        "rate_eq_0": (sum(v == 0 for v in valid) / n) if n else None,
                        "mean": stats["mean"],
                        "median": stats["median"],
                        "std": stats["std"],
                        "ci95_low": ci["ci95_low"],
                        "ci95_high": ci["ci95_high"],
                    }
                )
    return rows


def latency_rows(generation: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for subset in ("supported", "unsupported", "all"):
        ids = _subset_ids(subset)
        seconds = [generation[qid]["total_duration_ms"] / 1000.0 for qid in ids]
        stats = latency_describe(seconds)
        ci = bootstrap_mean_ci(seconds, seed_key=f"phase2_latency_s:{subset}")
        rows.append(
            {
                "record_type": "latency_seconds",
                "subset": subset,
                "dimension": "total_duration_s",
                "score": "",
                "count": stats["n"],
                "percentage": None,
                "n_valid": stats["n"],
                "n_expected": len(ids),
                "n_missing": len(ids) - stats["n"],
                "mean": stats["mean"],
                "median": stats["median"],
                "std": stats["std"],
                "min": stats["min"],
                "max": stats["max"],
                "p90": stats["p90"],
                "p95": stats["p95"],
                "range": stats["range"],
                "ci95_low": ci["ci95_low"],
                "ci95_high": ci["ci95_high"],
            }
        )
    return rows


def unsupported_case_rows(
    human: dict[str, dict[str, Any]],
    generation: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for qid in UNSUPPORTED_IDS:
        gen = generation[qid]
        abstention = gen.get("abstention") or {}
        rows.append(
            {
                "record_type": "unsupported_case",
                "subset": "unsupported",
                "dimension": "unsupported_handling",
                "question_id": qid,
                "insufficient_evidence": bool(gen["insufficient_evidence"]),
                "has_abstention_phrase": bool(abstention.get("has_abstention_phrase")),
                "abstention_phrases": " | ".join(abstention.get("abstention_phrase_matched") or []),
                "human_unsupported_handling": human[qid]["human_unsupported_handling"],
                "human_archive_correctness": human[qid]["human_archive_correctness"],
                "human_faithfulness": human[qid]["human_faithfulness"],
                "n_sources": len(gen.get("sources") or []),
            }
        )
    return rows


def agreement_rows(
    human: dict[str, dict[str, Any]],
    auto_scores: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary: list[dict[str, Any]] = []
    divergences: list[dict[str, Any]] = []
    provenance_filters = ("all_valid", "original_json_judgment", "recovered_from_original_raw_response", "second_evaluation")
    for human_key, auto_key in AGREEMENT_DIMENSIONS:
        for provenance in provenance_filters:
            pairs: list[tuple[str, int, int, str]] = []
            for qid in EXPECTED_IDS:
                auto = auto_scores.get(qid)
                if auto is None or auto.get(auto_key) is None:
                    continue
                if provenance != "all_valid" and auto.get("provenance") != provenance:
                    continue
                human_score = human[qid][human_key]
                if human_score is None:
                    continue
                pairs.append((qid, int(human_score), int(auto[auto_key]), str(auto.get("provenance"))))
            human_vals = [item[1] for item in pairs]
            auto_vals = [item[2] for item in pairs]
            n = len(pairs)
            exact = (
                sum(h == a for h, a in zip(human_vals, auto_vals, strict=True)) / n
                if n
                else None
            )
            mae = (
                sum(abs(h - a) for h, a in zip(human_vals, auto_vals, strict=True)) / n
                if n
                else None
            )
            kappa = quadratic_weighted_kappa(human_vals, auto_vals) if n else None
            higher = sum(a > h for h, a in zip(human_vals, auto_vals, strict=True))
            lower = sum(a < h for h, a in zip(human_vals, auto_vals, strict=True))
            row = {
                "dimension": auto_key,
                "provenance": provenance,
                "n_pairs": n,
                "exact_agreement": exact,
                "mean_absolute_difference": mae,
                "quadratic_weighted_kappa": kappa,
                "n_auto_higher": higher if n else None,
                "n_auto_lower": lower if n else None,
                "n_equal": (n - higher - lower) if n else None,
                "kappa_status": "computed" if kappa is not None else "undefined_or_not_calculable",
            }
            if n:
                row.update(confusion_counts(human_vals, auto_vals))
            summary.append(row)
            if provenance == "all_valid":
                for qid, h, a, prov in pairs:
                    if h != a:
                        divergences.append(
                            {
                                "question_id": qid,
                                "dimension": auto_key,
                                "human": h,
                                "automatic": a,
                                "automatic_minus_human": a - h,
                                "absolute_difference": abs(a - h),
                                "provenance": prov,
                                "auto_higher": a > h,
                            }
                        )
    return summary, divergences


def error_case_rows(
    *,
    human: dict[str, dict[str, Any]],
    generation: dict[str, dict[str, Any]],
    citation_errors: list[dict[str, Any]],
    phase1_a5: dict[str, dict[str, Any]],
    gold_docs: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    by_cit = {row["question_id"]: row for row in citation_errors}
    rows: list[dict[str, Any]] = []
    for qid in EXPECTED_IDS:
        cit = by_cit[qid]
        gen = generation[qid]
        if cit["has_wrong_document"]:
            rows.append(
                _case(qid, "wrong_document", human, gen, {"citations": cit["wrong_document_citations"]})
            )
        if cit["has_page_mismatch"]:
            rows.append(
                _case(
                    qid,
                    "page_differs_from_gold",
                    human,
                    gen,
                    {
                        "note": "Page difference versus gold pdf_pages. Not classified as hallucination.",
                        "mismatches": cit["page_mismatches_vs_gold"],
                    },
                )
            )
        archive = human[qid]["human_archive_correctness"]
        if archive in (0, 1):
            rows.append(
                _case(qid, "human_archive_correctness_0_or_1", human, gen, {"score": archive})
            )
        abstention = gen.get("abstention") or {}
        if (not gen["unsupported"]) and (
            gen["insufficient_evidence"] or abstention.get("has_abstention_phrase")
        ):
            rows.append(
                _case(
                    qid,
                    "abstention_on_supported_query",
                    human,
                    gen,
                    {
                        "insufficient_evidence": gen["insufficient_evidence"],
                        "phrases": abstention.get("abstention_phrase_matched") or [],
                    },
                )
            )
        doc_p = (gen.get("citation_metrics") or {}).get("citation_doc_precision")
        faith = human[qid]["human_faithfulness"]
        if doc_p == 1.0 and (
            (archive in (0, 1)) or (faith in (0, 1))
        ):
            rows.append(
                _case(
                    qid,
                    "document_aligned_citations_with_human_score_below_2",
                    human,
                    gen,
                    {
                        "citation_doc_precision": doc_p,
                        "human_archive_correctness": archive,
                        "human_faithfulness": faith,
                    },
                )
            )
    rows.append(_q021_case(human, generation, phase1_a5, gold_docs))
    return rows


def _case(qid, case_type, human, gen, detail) -> dict[str, Any]:
    return {
        "question_id": qid,
        "case_type": case_type,
        "unsupported": gen["unsupported"],
        "human_archive_correctness": human[qid]["human_archive_correctness"],
        "human_faithfulness": human[qid]["human_faithfulness"],
        "human_answer_relevance": human[qid]["human_answer_relevance"],
        "human_citation_semantic_support": human[qid]["human_citation_semantic_support"],
        "citation_doc_precision": (gen.get("citation_metrics") or {}).get("citation_doc_precision"),
        "insufficient_evidence": gen["insufficient_evidence"],
        "detail": detail,
    }


def _q021_case(human, generation, phase1_a5, gold_docs) -> dict[str, Any]:
    qid = "Q021"
    gen = generation[qid]
    retrieval = phase1_a5[qid]
    gold_ids = [str(item) for item in retrieval["primary_relevant"]]
    ranked = list(retrieval["ranked_pfc_ids"])
    cited = [str(src.get("pfc_id")) for src in gen.get("sources") or []]
    evidence_ids = [str(item.get("pfc_id")) for item in gen.get("evidence_items") or []]
    detail = {
        "gold_primary": [
            {
                "pfc_id": pfc_id,
                "year": (gold_docs.get(pfc_id) or {}).get("year"),
                "author": (gold_docs.get(pfc_id) or {}).get("author"),
                "title": (gold_docs.get(pfc_id) or {}).get("title"),
                "in_a5_ranking": pfc_id in ranked,
                "a5_rank": (ranked.index(pfc_id) + 1) if pfc_id in ranked else None,
            }
            for pfc_id in gold_ids
        ],
        "a5_mrr": retrieval["mrr"],
        "a5_recall_at_5": retrieval["recall_at_5"],
        "a5_recall_at_10": retrieval["recall_at_10"],
        "cited_pfc_ids": cited,
        "evidence_pfc_ids": evidence_ids,
        "answer_excerpt": (gen.get("answer") or "")[:400],
        "interpretation_rule": (
            "Temporal retrieval/filter failure is reported only when the 2015 gold "
            "documents are absent from the A5 ranking or absent from the evidence pack "
            "while the answer denies 2015 projects."
        ),
    }
    gold_in_ranking = all(item["in_a5_ranking"] for item in detail["gold_primary"])
    gold_in_evidence = all(pfc_id in evidence_ids for pfc_id in gold_ids)
    detail["gold_documents_all_in_a5_ranking"] = gold_in_ranking
    detail["gold_documents_all_in_evidence_pack"] = gold_in_evidence
    detail["records_support_retrieval_or_filter_failure"] = (not gold_in_ranking) or (
        not gold_in_evidence
    )
    return _case(qid, "q021_temporal_check", human, gen, detail)
