"""Build the Chapter 5 statistical tables from frozen benchmark artefacts.

Example:
  uv run --group analysis python evaluation/run_statistical_analysis.py
"""

from __future__ import annotations

import json
import math
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.stats_lib import (  # noqa: E402
    CONFIGS,
    EXPECTED_IDS,
    MASTER_SEED,
    N_BOOT,
    VALIDATED_HUMAN_REVIEW,
    AnalysisError,
    index_phase1,
    load_human_review,
    load_jsonl,
    phase1_config_rows,
    phase1_paired,
    require_validated_archive_scores,
    sha256_file,
    verify_stored_ir_metrics,
    write_csv,
)
from evaluation.stats_phase2 import (  # noqa: E402
    agreement_rows,
    error_case_rows,
    human_stat_rows,
    latency_rows,
    load_generation,
    score_table,
    unsupported_case_rows,
)

PHASE1 = ROOT / "evaluation/results/runs/20261019T201422Z-f2713630"
PHASE2 = ROOT / "evaluation/results/runs/20261019T210644Z-d128e825"
HUMAN_REVIEW = ROOT / VALIDATED_HUMAN_REVIEW
ARCHIVE_MEAN_TARGETS = {
    "supported": 1.250,
    "unsupported": 2.000,
    "all": 38 / 28,
}
PLOT_METRIC_LABELS = {
    "precision_at_5": "Precision@5",
    "recall_at_5": "Recall@5",
    "recall_at_10": "Recall@10",
    "mrr": "MRR",
}
PLOT_DIMENSION_LABELS = {
    "human_faithfulness": "Fidelidade factual",
    "human_answer_relevance": "Relevância da resposta",
    "human_citation_semantic_support": "Suporte semântico das citações",
    "human_archive_correctness": "Correcção face ao arquivo",
    "faithfulness": "Fidelidade factual",
    "answer_relevance": "Relevância da resposta",
    "citation_semantic_support": "Suporte semântico das citações",
}
PLOT_CONFIG_LABELS = {
    "A1": "A1 BM25",
    "A2": "A2 Densa",
    "A3": "A3 Híbrida",
    "A4": "A4 Regras",
    "A5": "A5 Regras e LLM",
}
OUT = ROOT / "evaluation/results/statistical_analysis_20261020"


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None or value == "":
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _rows(records: list[dict[str, Any]], **filters: Any) -> list[dict[str, Any]]:
    selected = []
    for record in records:
        if all(record.get(key) == value for key, value in filters.items()):
            selected.append(record)
    return selected


def _one(records: list[dict[str, Any]], **filters: Any) -> dict[str, Any]:
    found = _rows(records, **filters)
    if len(found) != 1:
        raise AnalysisError(f"Expected one row for {filters}, found {len(found)}")
    return found[0]


def build_report(bundle: dict[str, Any]) -> str:
    config_rows = bundle["config_rows"]
    paired = bundle["paired_rows"]
    notes = bundle["paired_notes"]
    human_rows = bundle["human_rows"]
    agreement = bundle["agreement_rows"]
    divergences = bundle["divergences"]
    errors = bundle["error_rows"]
    lines = [
        "# Análise estatística das Fases 1 e 2",
        "",
        "Relatório numérico para uso posterior no Capítulo 5. Não confirma nem rejeita H0–H4.",
        "A revisão humana foi feita pelo próprio investigador. Não há segundo avaliador humano e não se calcula concordância entre avaliadores.",
        "O mesmo modelo local (`llama3.1:8b`) gerou as respostas e produziu os juízos automáticos.",
        "",
        "## Proveniência e método",
        "",
        f"- Fase 1: `{PHASE1.relative_to(ROOT).as_posix()}`",
        f"- Fase 2: `{PHASE2.relative_to(ROOT).as_posix()}`",
        f"- Reavaliação: `{(PHASE2 / 'reevaluation').relative_to(ROOT).as_posix()}`",
        f"- Revisão humana: `{bundle['human_path'].relative_to(ROOT).as_posix()}` (folha `{bundle['human_meta']['sheet']}`)",
        f"- Bootstrap: {N_BOOT} reamostragens, IC percentil 95% (quantis 2,5% e 97,5%), semente-mestre {MASTER_SEED}.",
        "- Cada intervalo usa uma semente derivada por SHA-256 da chave da métrica, para a ordem de cálculo não alterar o resultado.",
        "- Desvio-padrão amostral com `ddof=1`. Percentis de latência: método linear do NumPy.",
        "- Métricas de recuperação: apenas consultas suportadas (n=24). Q023–Q026 ficam de fora desses denominadores.",
        "- Valores em falta permanecem em falta. Não entram nas médias.",
        "- Testes de Wilcoxon e a correcção de Holm são exploratórios. Não foram pré-registados como confirmação.",
        "- A única revisão humana aceite é `evaluation/human_review/revisao_manual_pfc_validada.xlsx`.",
        "- Antes do cálculo, confirmaram-se Q018 e Q023–Q026 com `human_archive_correctness` = 2.",
        "- As colunas `auto_*` da folha não são a fonte das pontuações automáticas.",
        "",
    ]
    lines.extend(
        [
            "### Precision@5",
            "",
            "A função `precision_at_k` divide o número de documentos relevantes entre os primeiros `k` pelo próprio `k` (k=5). O denominador não passa a ser o número de documentos devolvidos quando a lista é mais curta. Uma lista vazia vale 0. Esta é a regra já gravada em `retrieval.jsonl`; não foi alterada.",
            f"Pares configuração×consulta suportada com menos de 5 documentos distintos no ranking: {bundle['ir_check']['supported_rankings_shorter_than_5']}.",
            f"Divergências entre métricas guardadas e `score_ranking`: {len(bundle['ir_check']['mismatches'])}.",
            "",
            "## Fase 1 — configurações A1–A5",
            "",
            "Diferenças emparelhadas mais abaixo usam a ordem alternativa − base: A2−A1, A3−A1, A3−A2, A4−A3, A5−A3.",
            "",
            "| Config | P@5 | R@5 | R@10 | MRR | Latência média (ms) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for cfg in CONFIGS:
        cells = []
        for metric in ("precision_at_5", "recall_at_5", "recall_at_10", "mrr", "retrieval_duration_ms"):
            row = _one(config_rows, config_id=cfg, subset="supported", category="", metric=metric)
            cells.append(
                f"{_fmt(row['mean'])} [{_fmt(row['ci95_low'])}, {_fmt(row['ci95_high'])}]"
            )
        lines.append(f"| {cfg} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "Médias com IC bootstrap de 95% entre parênteses rectos. Denominador n=24 em todas as células.",
            "",
            "### Subgrupos do gold (consultas suportadas)",
            "",
        ]
    )
    categories = sorted(
        {
            row["category"]
            for row in config_rows
            if row["subset"] == "supported_category" and row["config_id"] == "A1"
        }
    )
    lines.append("| Categoria | n | " + " | ".join(f"{cfg} MRR" for cfg in CONFIGS) + " |")
    lines.append("|---|---:|" + "|".join(["---:" for _ in CONFIGS]) + "|")
    for category in categories:
        n = _one(
            config_rows,
            config_id="A1",
            subset="supported_category",
            category=category,
            metric="mrr",
        )["n"]
        means = [
            _fmt(
                _one(
                    config_rows,
                    config_id=cfg,
                    subset="supported_category",
                    category=category,
                    metric="mrr",
                )["mean"]
            )
            for cfg in CONFIGS
        ]
        lines.append(f"| {category} | {n} | " + " | ".join(means) + " |")
    lines.extend(["", "As quatro consultas `unsupported` não entram nestas médias.", ""])
    lines.extend(["## Comparações emparelhadas", ""])
    lines.append("| Contraste | Métrica | Diferença média | IC 95% | Melhorou | Piorou | Empate | p Wilcoxon | p Holm |")
    lines.append("|---|---|---:|---|---:|---:|---:|---:|---:|")
    for row in paired:
        if row["metric"] == "retrieval_duration_ms":
            continue
        lines.append(
            "| {comparison} | {metric} | {mean} | [{lo}, {hi}] | {imp} | {wor} | {tie} | {p} | {holm} |".format(
                comparison=row["comparison"],
                metric=row["metric"],
                mean=_fmt(row["mean_difference"]),
                lo=_fmt(row["ci95_low"]),
                hi=_fmt(row["ci95_high"]),
                imp=row["n_improved"],
                wor=row["n_worsened"],
                tie=row["n_tied"],
                p=_fmt(row["pvalue_unadjusted"], 4),
                holm=_fmt(row["pvalue_holm"], 4),
            )
        )
    identity = notes["a3_vs_a4"]
    focus = notes["a5_vs_a3_mrr"]
    lines.extend(
        [
            "",
            f"Família de Holm: {notes['holm_family_size']} testes de Wilcoxon aplicáveis (MRR, Recall@5, Recall@10). Contrastes com todas as diferenças nulas não recebem valor-p e ficam fora da correcção.",
            f"A3 e A4 têm métricas idênticas nas 28 consultas: {identity['metric_identical_all_28']}. Rankings idênticos: {identity['ranking_identical_all_28']}. IDs divergentes: {identity['differing_question_ids'] or 'nenhum'}.",
            f"MRR A5−A3 médio: {_fmt(focus['mean_mrr_difference'], 4)}. As únicas consultas com MRR diferente são Q009 ({_fmt(focus['changed_question_ids'].get('Q009'), 4)}) e Q011 ({_fmt(focus['changed_question_ids'].get('Q011'), 4)}). O sinal é oposto: Q011 sobe e Q009 desce. O ganho médio é o saldo destas duas diferenças. Outras consultas com MRR diferente: {focus['other_changed'] or 'nenhuma'}.",
            "",
            "Latências emparelhadas estão em `phase1_paired_comparisons.csv` e não entram na família de Holm.",
            "",
            "## Fase 2 — revisão humana",
            "",
            "Notas humanas 0/1/2. Percentagens usam o número de notas válidas do subconjunto, indicado na coluna `n_valid`.",
            "Nas quatro consultas não suportadas, a folha regista `N/A` em fidelidade, relevância e suporte semântico das citações. Essas células ficam de fora das médias. O tratamento de não suportadas e a correcção face ao arquivo estão preenchidos.",
            "",
        ]
    )
    lines.append("| Subconjunto | Dimensão | n válido | n em falta | %0 | %1 | %2 | Média | Mediana | DP | IC 95% |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    seen = []
    for row in human_rows:
        if row["record_type"] != "frequency" or row["score"] != 0:
            continue
        key = (row["subset"], row["dimension"])
        if key in seen:
            continue
        seen.append(key)
        freqs = {
            score: _one(
                human_rows,
                record_type="frequency",
                subset=row["subset"],
                dimension=row["dimension"],
                score=score,
            )
            for score in (0, 1, 2)
        }
        lines.append(
            "| {subset} | {dim} | {n} | {miss} | {p0} | {p1} | {p2} | {mean} | {med} | {std} | [{lo}, {hi}] |".format(
                subset=row["subset"],
                dim=row["dimension"],
                n=row["n_valid"],
                miss=row["n_missing"],
                p0=_fmt(freqs[0]["percentage"], 1),
                p1=_fmt(freqs[1]["percentage"], 1),
                p2=_fmt(freqs[2]["percentage"], 1),
                mean=_fmt(row["mean"]),
                med=_fmt(row["median"]),
                std=_fmt(row["std"]),
                lo=_fmt(row["ci95_low"]),
                hi=_fmt(row["ci95_high"]),
            )
        )
    lines.extend(["", "### Correcção face ao arquivo", ""])
    for row in human_rows:
        if row["record_type"] != "archive_rate":
            continue
        lines.append(
            f"- {row['subset']}: n={row['n_valid']} (em falta={row['n_missing']}); "
            f"média = {float(row['mean']):.9f}; "
            f"taxa =2: {_fmt(row['rate_eq_2'])}; taxa ≥1: {_fmt(row['rate_ge_1'])}; "
            f"taxa =0: {_fmt(row['rate_eq_0'])}."
        )
    lines.extend(["", "### Consultas não suportadas", ""])
    lines.append("| ID | insufficient_evidence | frase de abstenção | nota humana de tratamento | correcção arquivo |")
    lines.append("|---|---|---|---:|---:|")
    for row in human_rows:
        if row["record_type"] != "unsupported_case":
            continue
        lines.append(
            f"| {row['question_id']} | {row['insufficient_evidence']} | {row['has_abstention_phrase']} | "
            f"{_fmt(row['human_unsupported_handling'], 0)} | {_fmt(row['human_archive_correctness'], 0)} |"
        )
    latency = _one(human_rows, record_type="latency_seconds", subset="all")
    lines.extend(
        [
            "",
            "### Latência total da resposta (segundos, n=28)",
            "",
            f"Média {_fmt(latency['mean'])}; mediana {_fmt(latency['median'])}; DP {_fmt(latency['std'])}; "
            f"P90 {_fmt(latency['p90'])}; P95 {_fmt(latency['p95'])}; amplitude {_fmt(latency['range'])} "
            f"(mínimo {_fmt(latency['min'])}, máximo {_fmt(latency['max'])}).",
            "",
            "## Concordância humano / juiz automático",
            "",
            "Pares com as duas notas válidas. Q023 não tem pontuação automática completa e sai destes denominadores quando a nota automática falta.",
            "Kappa ponderado quadrático fica indefinido quando a discordância esperada é zero.",
            "",
            "| Dimensão | Proveniência | n | Concordância exacta | |Δ| médio | Kappa quadrático | Auto > humano | Auto < humano |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in agreement:
        if row.get("record_type", "agreement") != "agreement":
            continue
        lines.append(
            f"| {row['dimension']} | {row['provenance']} | {row['n_pairs']} | {_fmt(row['exact_agreement'])} | "
            f"{_fmt(row['mean_absolute_difference'])} | {_fmt(row['quadratic_weighted_kappa'])} | "
            f"{_fmt(row['n_auto_higher'], 0)} | {_fmt(row['n_auto_lower'], 0)} |"
        )
    high = [row for row in divergences if row["absolute_difference"] >= 2]
    q021 = [row for row in divergences if row["question_id"] == "Q021"]
    lines.extend(
        [
            "",
            "Divergências com |Δ|≥2: "
            + (
                ", ".join(
                    f"{row['question_id']} {row['dimension']} humano={row['human']} automático={row['automatic']}"
                    for row in high
                )
                or "nenhuma"
            )
            + ".",
            "Q021, juízo automático original: fidelidade, relevância e suporte semântico humanos = 0 e automáticos = 2.",
            "",
            "## Diagnóstico de casos",
            "",
            "Diferenças de página face ao gold não são classificadas como alucinação.",
            "",
        ]
    )
    by_type: dict[str, list[str]] = {}
    for row in errors:
        by_type.setdefault(row["case_type"], []).append(row["question_id"])
    for case_type, ids in by_type.items():
        lines.append(f"- {case_type}: {len(ids)} registos; IDs {sorted(set(ids))}.")
    q021_row = _one_case(errors, "q021_temporal_check")
    detail = q021_row["detail"]
    lines.extend(
        [
            "",
            "### Q021",
            "",
            "Os dois PFC gold de 2015 (Mayra Patrícia Duarte Sofiano; Diana João Matsinhe) estão ausentes do ranking A5 e do evidence pack da Fase 2.",
            f"MRR A5={_fmt(detail['a5_mrr'])}; Recall@5={_fmt(detail['a5_recall_at_5'])}; Recall@10={_fmt(detail['a5_recall_at_10'])}.",
            "Isto confirma uma falha de recuperação desses documentos na configuração A5. O ranking não regista um filtro de ano separado, pelo que a causa específica «filtragem temporal» não fica isolada pelos dados.",
            "",
            "## Limitações",
            "",
            "- Um único investigador atribuiu as notas humanas.",
            "- O juiz automático partilha o modelo de geração.",
            "- Várias pontuações automáticas foram recuperadas de texto ou obtidas numa segunda avaliação; a proveniência está separada na tabela de concordância.",
            "- Os testes múltiplos são exploratórios. Um IC que exclui zero, ou um valor-p baixo, não constitui confirmação pré-registada de hipótese.",
            "- Precision@5 com denominador fixo 5 comprime a escala quando há um ou poucos documentos relevantes.",
            "- Vários contrastes de Wilcoxon assentam em poucas diferenças não nulas.",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def _one_case(rows: list[dict[str, Any]], case_type: str) -> dict[str, Any]:
    found = [row for row in rows if row["case_type"] == case_type]
    if len(found) != 1:
        raise AnalysisError(f"Expected one {case_type}, found {len(found)}")
    return found[0]


def save_plots(out_dir: Path, bundle: dict[str, Any]) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written: list[str] = []
    config_rows = bundle["config_rows"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    metrics = ("precision_at_5", "recall_at_5", "recall_at_10", "mrr")
    width = 0.18
    for index, metric in enumerate(metrics):
        means, lows, highs = [], [], []
        for cfg in CONFIGS:
            row = _one(config_rows, config_id=cfg, subset="supported", category="", metric=metric)
            means.append(row["mean"])
            lows.append(row["mean"] - row["ci95_low"])
            highs.append(row["ci95_high"] - row["mean"])
        positions = [pos + (index - 1.5) * width for pos in range(len(CONFIGS))]
        ax.bar(
            positions,
            means,
            width=width,
            label=PLOT_METRIC_LABELS[metric],
            yerr=[lows, highs],
            capsize=2,
        )
    ax.set_xticks(range(len(CONFIGS)), [PLOT_CONFIG_LABELS[cfg] for cfg in CONFIGS])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Média nas consultas suportadas (n = 24)")
    ax.set_title("Fase 1 — média e intervalo de confiança bootstrap a 95%")
    ax.legend(title="Métrica", fontsize=8)
    fig.tight_layout()
    path = out_dir / "phase1_ir_means_ci.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(path.name)

    fig, ax = plt.subplots(figsize=(8, 4))
    data = []
    for cfg in CONFIGS:
        data.append(
            [
                bundle["phase1"][cfg][qid]["retrieval_duration_ms"]
                for qid in EXPECTED_IDS
                if not bundle["phase1"][cfg][qid]["unsupported"]
            ]
        )
    ax.boxplot(data, tick_labels=[PLOT_CONFIG_LABELS[cfg] for cfg in CONFIGS], showfliers=True)
    ax.set_yscale("log")
    ax.set_ylabel("Latência de recuperação (ms)")
    ax.set_title("Fase 1 — latência de recuperação nas consultas suportadas (n = 24)")
    fig.tight_layout()
    path = out_dir / "phase1_latency_box.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(path.name)

    human_rows = [
        row
        for row in bundle["human_rows"]
        if row["record_type"] == "frequency" and row["subset"] == "supported"
    ]
    dimensions = []
    for row in human_rows:
        if row["dimension"] not in dimensions:
            dimensions.append(row["dimension"])
    fig, ax = plt.subplots(figsize=(9, 4.5))
    import numpy as np

    x = np.arange(len(dimensions))
    for score, offset in ((0, -0.25), (1, 0.0), (2, 0.25)):
        counts = [
            _one(human_rows, dimension=dim, score=score)["count"] for dim in dimensions
        ]
        ax.bar(x + offset, counts, width=0.25, label=str(score))
    ax.set_xticks(
        x,
        [PLOT_DIMENSION_LABELS[dim] for dim in dimensions],
        rotation=18,
        ha="right",
    )
    ax.set_ylabel("Número de consultas")
    ax.set_title("Fase 2 — distribuição das notas do investigador (n = 24)")
    ax.legend(title="Nota")
    fig.tight_layout()
    path = out_dir / "phase2_human_scores_supported.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(path.name)

    agreement = [
        row
        for row in bundle["agreement_rows"]
        if row["provenance"] == "all_valid" and row["n_pairs"]
    ]
    if agreement:
        fig, axes = plt.subplots(1, len(agreement), figsize=(4 * len(agreement), 3.6))
        if len(agreement) == 1:
            axes = [axes]
        for ax, row in zip(axes, agreement, strict=True):
            matrix = np.array(
                [[row.get(f"h{i}_a{j}", 0) for j in (0, 1, 2)] for i in (0, 1, 2)],
                dtype=float,
            )
            image = ax.imshow(matrix, cmap="Blues", vmin=0)
            for i in range(3):
                for j in range(3):
                    ax.text(j, i, str(int(matrix[i, j])), ha="center", va="center")
            ax.set_xticks((0, 1, 2), ("0", "1", "2"))
            ax.set_yticks((0, 1, 2), ("0", "1", "2"))
            ax.set_xlabel("Juiz automático")
            ax.set_ylabel("Investigador")
            ax.set_title(PLOT_DIMENSION_LABELS[row["dimension"]])
            colorbar = fig.colorbar(image, ax=ax, fraction=0.046)
            colorbar.set_label("Consultas")
        fig.tight_layout()
        path = out_dir / "phase2_agreement_confusion.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path.name)
    return written


def resolve_human_workbook() -> Path:
    if not HUMAN_REVIEW.is_file():
        raise AnalysisError(
            "Validated human review workbook is missing: "
            f"{VALIDATED_HUMAN_REVIEW}. Analysis stopped. "
            "No other review file is accepted."
        )
    return HUMAN_REVIEW


def assert_archive_means(human_rows: list[dict[str, Any]]) -> None:
    for subset, target in ARCHIVE_MEAN_TARGETS.items():
        row = _one(human_rows, record_type="archive_rate", subset=subset)
        actual = row["mean"]
        if actual is None or not math.isclose(float(actual), target, rel_tol=0, abs_tol=1e-9):
            raise AnalysisError(
                f"human_archive_correctness mean for {subset} is {actual}, expected {target}."
            )


def main() -> int:
    try:
        human_path = resolve_human_workbook()
        human, human_meta = load_human_review(human_path)
        require_validated_archive_scores(human)
    except AnalysisError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(
        "Confirmed human_archive_correctness: "
        + ", ".join(f"{qid}=2" for qid in ("Q018", "Q023", "Q024", "Q025", "Q026")),
        flush=True,
    )
    phase1_rows = load_jsonl(PHASE1 / "retrieval.jsonl")
    phase1 = index_phase1(phase1_rows)
    ir_check = verify_stored_ir_metrics(phase1)
    if ir_check["mismatches"]:
        raise AnalysisError(f"Stored IR metrics diverge: {ir_check['mismatches'][:10]}")
    config_rows = phase1_config_rows(phase1)
    paired_rows, query_rows, paired_notes = phase1_paired(phase1)
    generation = load_generation(PHASE2 / "generation.jsonl")
    coverage = json.loads((PHASE2 / "reevaluation" / "judgment_coverage.json").read_text(encoding="utf-8"))
    auto_scores = score_table(coverage)
    citation_errors = json.loads(
        (PHASE2 / "reevaluation" / "citation_error_analysis.json").read_text(encoding="utf-8")
    )
    gold = json.loads((ROOT / "evaluation/gold_questions.json").read_text(encoding="utf-8"))
    gold_ids = [str(item["id"]) for item in gold["questions"]]
    if gold_ids != EXPECTED_IDS:
        raise AnalysisError("Gold question ids are not Q001–Q028 in order")
    gold_docs = {doc["pfc_id"]: doc for doc in gold["documents"]}
    human_rows = (
        human_stat_rows(human)
        + unsupported_case_rows(human, generation)
        + latency_rows(generation)
    )
    assert_archive_means(human_rows)
    agreement, divergences = agreement_rows(human, auto_scores)
    for row in agreement:
        row["record_type"] = "agreement"
    for row in divergences:
        row["record_type"] = "divergence"
    errors = error_case_rows(
        human=human,
        generation=generation,
        citation_errors=citation_errors,
        phase1_a5=phase1["A5"],
        gold_docs=gold_docs,
    )
    bundle = {
        "config_rows": config_rows,
        "paired_rows": paired_rows,
        "paired_notes": paired_notes,
        "human_rows": human_rows,
        "human_meta": human_meta,
        "agreement_rows": agreement,
        "divergences": divergences,
        "error_rows": errors,
        "ir_check": ir_check,
        "phase1": phase1,
        "human_path": human_path,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    # Phase 1 tables stay as previously written. Only human-dependent outputs are replaced.
    write_csv(OUT / "phase2_human_stats.csv", human_rows)
    write_csv(OUT / "phase2_judge_agreement.csv", agreement + divergences)
    write_csv(OUT / "phase2_error_cases.csv", errors)
    plots = save_plots(OUT, bundle)
    report = build_report(bundle)
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "numpy": version("numpy"),
        "scipy": version("scipy"),
        "matplotlib": version("matplotlib"),
        "openpyxl": version("openpyxl"),
        "master_seed": MASTER_SEED,
        "n_bootstrap": N_BOOT,
        "bootstrap": "percentile 95% CI of the mean; query resampling with replacement",
        "std": "sample standard deviation, ddof=1",
        "percentiles": "numpy.quantile method=linear",
        "precision_at_5": ir_check["precision_at_5_rule"],
        "retrieval_denominator": "supported queries only, n=24",
        "paired_difference": "alternative minus baseline",
        "wilcoxon": "two-sided, zero differences excluded, scipy method=auto",
        "multiple_testing": "Holm on applicable Wilcoxon tests for MRR, Recall@5 and Recall@10; exploratory",
        "human_reviewer": "single investigator, not an independent second rater",
        "inter_rater_human_kappa": "not computed",
        "generation_and_judge_model": "llama3.1:8b",
        "hypotheses_H0_H4": "not confirmed or rejected by this analysis",
        "commands": [
            "uv run --group analysis python evaluation/run_statistical_analysis.py",
            "uv run python src/scripts/test_statistical_analysis.py",
        ],
        "sources": {
            str(path.relative_to(ROOT)).replace("\\", "/"): sha256_file(path)
            for path in (
                PHASE1 / "retrieval.jsonl",
                PHASE1 / "summary.json",
                PHASE1 / "run_manifest.json",
                PHASE2 / "generation.jsonl",
                PHASE2 / "judgments.jsonl",
                PHASE2 / "summary.json",
                PHASE2 / "reevaluation" / "judgment_coverage.json",
                PHASE2 / "reevaluation" / "citation_error_analysis.json",
                human_path,
                ROOT / "evaluation/gold_questions.json",
            )
        },
        "human_workbook": str(human_path.relative_to(ROOT)).replace("\\", "/"),
        "human_workbook_sha256": sha256_file(human_path),
        "confirmed_archive_scores": {
            "Q018": 2,
            "Q023": 2,
            "Q024": 2,
            "Q025": 2,
            "Q026": 2,
        },
        "human_archive_correctness_means": {
            "supported_n24": 1.250,
            "unsupported_n4": 2.000,
            "all_n28": 38 / 28,
        },
        "human_workbook_note": (
            "Only evaluation/human_review/revisao_manual_pfc_validada.xlsx is accepted. "
            "Spreadsheet auto_* columns are not the automatic scores; those come from "
            "reevaluation/judgment_coverage.json. Phase 1 CSV files were not rewritten."
        ),
        "human_sheet": human_meta,
        "plots": plots,
        "ir_integrity": {
            "mismatches": ir_check["mismatches"],
            "supported_rankings_shorter_than_5": ir_check["supported_rankings_shorter_than_5"],
        },
        "a3_vs_a4": paired_notes["a3_vs_a4"],
        "a5_vs_a3_mrr": paired_notes["a5_vs_a3_mrr"],
    }
    (OUT / "analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
