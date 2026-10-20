"""Integrity checks for the frozen-artefact statistical analysis.

Run:
  uv run python src/scripts/test_statistical_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evaluation.metrics.retrieval import precision_at_k
from evaluation.stats_lib import (
    EXPECTED_IDS,
    VALIDATED_HUMAN_REVIEW,
    AnalysisError,
    bootstrap_mean_ci,
    describe,
    holm_adjust,
    index_phase1,
    load_human_review,
    load_jsonl,
    parse_score,
    quadratic_weighted_kappa,
    require_validated_archive_scores,
    verify_stored_ir_metrics,
    wilcoxon_paired,
)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_precision_denominator_is_k() -> None:
    _assert(precision_at_k(["doc-a"], ["doc-a"], 5) == 0.2, "One hit in a short list is 1/5.")
    _assert(precision_at_k([], ["doc-a"], 5) == 0.0, "Empty ranking precision is 0.")
    _assert(precision_at_k(["x", "doc-a"], ["doc-a"], 5) == 0.2, "Hits are divided by k=5.")


def test_describe_skips_missing() -> None:
    stats = describe([1.0, None, 3.0])
    _assert(stats["n"] == 2, "Missing values stay out of the denominator.")
    _assert(stats["mean"] == 2.0, "Mean uses only observed values.")
    _assert(describe([None])["mean"] is None, "All-missing mean is missing, not zero.")


def test_bootstrap_is_reproducible() -> None:
    first = bootstrap_mean_ci([0.0, 1.0, 0.5, 1.0], seed_key="unit")
    second = bootstrap_mean_ci([0.0, 1.0, 0.5, 1.0], seed_key="unit")
    _assert(first == second, "The same seed key repeats the interval.")
    _assert(first["ci95_low"] <= first["ci95_high"], "CI bounds are ordered.")


def test_holm_monotone() -> None:
    adjusted = holm_adjust([("a", 0.01), ("b", 0.04), ("c", 0.03)])
    _assert(adjusted["a"] <= adjusted["c"] <= adjusted["b"], "Holm p-values stay monotone.")
    _assert(all(0 <= value <= 1 for value in adjusted.values()), "Adjusted p-values stay in [0, 1].")


def test_wilcoxon_all_ties() -> None:
    result = wilcoxon_paired([0.0, 0.0, 0.0])
    _assert(result["applicable"] is False, "Identical pairs do not receive a p-value.")
    _assert(result["pvalue"] is None, "No invented p-value for an inapplicable test.")


def test_kappa() -> None:
    perfect = quadratic_weighted_kappa([0, 1, 2, 0], [0, 1, 2, 0])
    _assert(perfect == 1.0, "Identical mixed ratings have kappa 1.")
    _assert(
        quadratic_weighted_kappa([2, 2, 2], [2, 2, 2]) is None,
        "Kappa is undefined when expected disagreement is zero.",
    )


def test_parse_score() -> None:
    _assert(parse_score("") is None, "Blank human scores stay missing.")
    _assert(parse_score("N/A") is None, "NA is missing.")
    _assert(parse_score(2) == 2, "Integer scores pass through.")
    try:
        parse_score(1.5)
    except AnalysisError:
        return
    raise AssertionError("Fractional human scores must be rejected.")


def _blank_review() -> dict:
    return {
        qid: {
            "human_faithfulness": None,
            "human_answer_relevance": None,
            "human_citation_semantic_support": None,
            "human_unsupported_handling": None,
            "human_archive_correctness": 2,
        }
        for qid in EXPECTED_IDS
    }


def test_outdated_archive_scores_are_rejected() -> None:
    human = _blank_review()
    human["Q018"]["human_archive_correctness"] = 1
    for qid in ("Q023", "Q024", "Q025", "Q026"):
        human[qid]["human_archive_correctness"] = 1
    try:
        require_validated_archive_scores(human)
    except AnalysisError as exc:
        _assert("Q018" in str(exc), "The rejection names the outdated question.")
        return
    raise AssertionError("An outdated human review was accepted.")


def test_confirmed_archive_scores_pass() -> None:
    require_validated_archive_scores(_blank_review())


def test_runner_accepts_only_validated_workbook() -> None:
    source = (ROOT / "evaluation/run_statistical_analysis.py").read_text(encoding="utf-8")
    _assert("HUMAN_CANDIDATES" not in source, "The runner must not keep a fallback workbook.")
    _assert(VALIDATED_HUMAN_REVIEW in source, "The runner must name the validated workbook.")
    _assert(
        "No other review file is accepted." in source,
        "A missing validated workbook must stop the analysis.",
    )


def test_validated_workbook_matches_confirmed_means() -> None:
    path = ROOT / VALIDATED_HUMAN_REVIEW
    _assert(path.is_file(), f"Validated workbook missing: {path}")
    human, _meta = load_human_review(path)
    require_validated_archive_scores(human)
    supported = [
        human[qid]["human_archive_correctness"]
        for qid in EXPECTED_IDS
        if qid not in {"Q023", "Q024", "Q025", "Q026"}
    ]
    unsupported = [
        human[qid]["human_archive_correctness"]
        for qid in ("Q023", "Q024", "Q025", "Q026")
    ]
    overall = [human[qid]["human_archive_correctness"] for qid in EXPECTED_IDS]
    _assert(len(supported) == 24 and all(score is not None for score in supported), "Supported scores")
    _assert(sum(supported) / 24 == 1.25, "Supported archive mean")
    _assert(sum(unsupported) / 4 == 2.0, "Unsupported archive mean")
    _assert(abs(sum(overall) / 28 - (38 / 28)) < 1e-12, "Overall archive mean")


def test_previous_workbook_is_rejected() -> None:
    old = ROOT / "evaluation/revisao_manual_pfc.xlsx"
    if not old.is_file():
        return
    human, _meta = load_human_review(old)
    try:
        require_validated_archive_scores(human)
    except AnalysisError:
        return
    raise AssertionError("The previous review workbook was accepted.")


def test_phase1_artefact_integrity() -> None:
    path = ROOT / "evaluation/results/runs/20261019T201422Z-f2713630/retrieval.jsonl"
    grouped = index_phase1(load_jsonl(path))
    for config, rows in grouped.items():
        _assert(list(rows) == EXPECTED_IDS or set(rows) == set(EXPECTED_IDS), config)
        supported = [row for row in rows.values() if not row["unsupported"]]
        _assert(len(supported) == 24, f"{config} supported denominator")
        _assert(sum(1 for row in rows.values() if row["unsupported"]) == 4, f"{config} unsupported")
    check = verify_stored_ir_metrics(grouped)
    _assert(check["mismatches"] == [], f"Stored metrics diverged: {check['mismatches'][:5]}")


def main() -> int:
    test_precision_denominator_is_k()
    test_describe_skips_missing()
    test_bootstrap_is_reproducible()
    test_holm_monotone()
    test_wilcoxon_all_ties()
    test_kappa()
    test_parse_score()
    test_outdated_archive_scores_are_rejected()
    test_confirmed_archive_scores_pass()
    test_runner_accepts_only_validated_workbook()
    test_validated_workbook_matches_confirmed_means()
    test_previous_workbook_is_rejected()
    test_phase1_artefact_integrity()
    print("Statistical analysis integrity checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
