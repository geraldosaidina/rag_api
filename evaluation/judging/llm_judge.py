"""LLM-as-judge for Phase 2 faithfulness / relevance / citation support.

Records model name, prompts, scale, and flags cases for human review.
Does not treat gold answer_notes as canonical answers.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from rag.llm.ollama_client import LLMException, OllamaLLMClient

JUDGE_SCALE = {
    "faithfulness": "0 = unsupported claims; 1 = mostly supported; 2 = fully supported by evidence pack",
    "answer_relevance": "0 = off-topic; 1 = partial; 2 = addresses the question or correctly abstains",
    "citation_semantic_support": (
        "0 = citations do not support claims; 1 = mixed; "
        "2 = cited fragments support adjacent claims"
    ),
    "unsupported_handling": (
        "Only for unsupported gold questions. "
        "0 = affirms false/absent premise; 1 = hedged; "
        "2 = refuses or says retrieved evidence does not establish the request"
    ),
}

SYSTEM_PROMPT = """You are an evaluation judge for a university RAG assistant over final-course projects (PFCs).

Rules:
1. Score ONLY using the supplied evidence pack and the assistant answer.
2. Do NOT use outside knowledge.
3. Do NOT treat any "answer notes" as a required wording; they are optional hints only.
4. Faithfulness means every material factual claim is supported by the evidence pack.
5. Citation semantic support means cited labels point to fragments that support the nearby claims.
6. For unsupported questions, reward correct refusal / non-establishment; punish affirming the false premise.
7. Return ONLY valid JSON with the required keys.
"""


@dataclass(frozen=True)
class JudgeResult:
    faithfulness: int | None
    answer_relevance: int | None
    citation_semantic_support: int | None
    unsupported_handling: int | None
    rationale: str
    needs_human_review: bool
    human_review_reasons: list[str]
    raw_response: str
    parse_error: str | None
    model_name: str


def build_user_prompt(
    *,
    question: str,
    answer: str,
    evidence_context_text: str,
    sources: list[dict[str, Any]],
    unsupported: bool,
    answer_notes: str | None,
) -> str:
    notes_block = (
        f"Optional factual hints for the judge (NOT a required answer wording):\n{answer_notes}\n\n"
        if answer_notes
        else ""
    )
    return (
        f"Question:\n{question}\n\n"
        f"Assistant answer:\n{answer}\n\n"
        f"Evidence pack supplied to the model:\n{evidence_context_text or '(empty)'}\n\n"
        f"Structured sources cited after validation:\n{json.dumps(sources, ensure_ascii=False)}\n\n"
        f"Gold marks this question unsupported: {unsupported}\n\n"
        f"{notes_block}"
        "Return JSON with keys:\n"
        "faithfulness (0-2),\n"
        "answer_relevance (0-2),\n"
        "citation_semantic_support (0-2),\n"
        "unsupported_handling (0-2 or null if not unsupported),\n"
        "rationale (short Portuguese string),\n"
        "needs_human_review (boolean),\n"
        "human_review_reasons (array of short strings).\n"
    )


def _extract_json(raw: str) -> dict[str, Any]:
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
        raise ValueError("Judge response is not a JSON object.")
    return parsed


def _clamp_score(value: Any) -> int | None:
    if value is None:
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    if number < 0 or number > 2:
        return None
    return number


def judge_answer(
    llm_client: OllamaLLMClient,
    *,
    question: str,
    answer: str,
    evidence_context_text: str,
    sources: list[dict[str, Any]],
    unsupported: bool,
    answer_notes: str | None,
) -> JudgeResult:
    user_prompt = build_user_prompt(
        question=question,
        answer=answer,
        evidence_context_text=evidence_context_text,
        sources=sources,
        unsupported=unsupported,
        answer_notes=answer_notes,
    )
    try:
        response = llm_client.invoke_messages(
            [("system", SYSTEM_PROMPT), ("user", user_prompt)]
        )
        raw = response.content
        model_name = response.model_name
    except LLMException as exc:
        return JudgeResult(
            faithfulness=None,
            answer_relevance=None,
            citation_semantic_support=None,
            unsupported_handling=None,
            rationale="",
            needs_human_review=True,
            human_review_reasons=[f"llm_invoke_failed: {exc}"],
            raw_response="",
            parse_error=str(exc),
            model_name=llm_client.config.model_name,
        )

    try:
        parsed = _extract_json(raw)
        faithfulness = _clamp_score(parsed.get("faithfulness"))
        relevance = _clamp_score(parsed.get("answer_relevance"))
        citation_support = _clamp_score(parsed.get("citation_semantic_support"))
        unsupported_handling = _clamp_score(parsed.get("unsupported_handling"))
        if not unsupported:
            unsupported_handling = None
        rationale = str(parsed.get("rationale") or "").strip()
        needs_human = bool(parsed.get("needs_human_review"))
        reasons = parsed.get("human_review_reasons") or []
        if not isinstance(reasons, list):
            reasons = [str(reasons)]
        reasons = [str(item) for item in reasons]

        if None in (faithfulness, relevance, citation_support):
            needs_human = True
            reasons.append("missing_or_invalid_score")
        if unsupported and unsupported_handling is None:
            needs_human = True
            reasons.append("missing_unsupported_handling_score")
        if any(score == 1 for score in (faithfulness, relevance, citation_support) if score is not None):
            needs_human = True
            reasons.append("mid_score")
        if unsupported and unsupported_handling in {0, 1}:
            needs_human = True
            reasons.append("unsupported_not_clearly_handled")

        return JudgeResult(
            faithfulness=faithfulness,
            answer_relevance=relevance,
            citation_semantic_support=citation_support,
            unsupported_handling=unsupported_handling,
            rationale=rationale,
            needs_human_review=needs_human,
            human_review_reasons=reasons,
            raw_response=raw,
            parse_error=None,
            model_name=model_name,
        )
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        return JudgeResult(
            faithfulness=None,
            answer_relevance=None,
            citation_semantic_support=None,
            unsupported_handling=None,
            rationale="",
            needs_human_review=True,
            human_review_reasons=[f"parse_error: {exc}"],
            raw_response=raw,
            parse_error=str(exc),
            model_name=model_name,
        )


def judge_prompt_record(model_name: str) -> dict[str, Any]:
    return {
        "judge_model_name": model_name,
        "system_prompt": SYSTEM_PROMPT,
        "scale": JUDGE_SCALE,
        "output_schema": [
            "faithfulness",
            "answer_relevance",
            "citation_semantic_support",
            "unsupported_handling",
            "rationale",
            "needs_human_review",
            "human_review_reasons",
        ],
        "decision_criteria": (
            "Scores use only the evidence pack and assistant answer. "
            "Mid scores, unsupported failures, parse errors, and model errors "
            "require human review."
        ),
    }
