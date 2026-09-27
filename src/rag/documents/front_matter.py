"""Resumo and keyword extraction from PFC page text."""

from __future__ import annotations

import re

_MAX_ABSTRACT_CHARS = 3000
_KEYWORDS = re.compile(r"palavras\s*[-–—]?\s*chaves?\s*[:.]?\s*(.*)", re.IGNORECASE)
_STOP_HEADING = re.compile(
    r"^(palavras\b|abstract\b|introdu|índice\b|indice\b|cap[ií]tulo\b|"
    r"lista de\b|agradec|dedicat|declara|sum[aá]rio\b)",
    re.IGNORECASE,
)


def extract_abstract(pages: list[str]) -> str | None:
    best = ""
    for page in pages:
        lines = _text_lines(page)
        for index, line in enumerate(lines):
            inline = _inline_resumo(line)
            if inline is None and not _is_resumo_heading(line):
                continue
            body: list[str] = [inline] if inline else []
            for follow in lines[index + 1 :]:
                if not follow:
                    if body:
                        break
                    continue
                if (
                    _KEYWORDS.search(follow)
                    or _STOP_HEADING.match(follow)
                    or _is_resumo_heading(follow)
                ):
                    break
                body.append(follow)
            candidate = " ".join(part for part in body if part).strip()
            if len(candidate) > len(best):
                best = candidate
    if len(best) < 80:
        return None
    return best[:_MAX_ABSTRACT_CHARS].strip()


def extract_keywords(pages: list[str]) -> tuple[str, ...]:
    for page in pages:
        lines = _text_lines(page)
        for index, line in enumerate(lines):
            match = _KEYWORDS.search(line)
            if not match:
                continue
            parts = [match.group(1).strip()]
            for follow in lines[index + 1 : index + 5]:
                if not follow or _STOP_HEADING.match(follow) or _is_resumo_heading(follow):
                    if parts and any(parts):
                        break
                    continue
                if _KEYWORDS.search(follow):
                    break
                parts.append(follow)
            keywords = _split_keywords(" ".join(part for part in parts if part))
            if keywords:
                return keywords
    return ()


def _text_lines(text: str) -> list[str]:
    normalized = text.replace("\u00a0", " ").replace("\r", "\n")
    return [re.sub(r"[ \t]+", " ", line).strip() for line in normalized.splitlines()]


def _inline_resumo(line: str) -> str | None:
    """Resumo body when the heading and paragraph share one extracted line."""
    match = re.search(r"\bresumo\b", line, flags=re.IGNORECASE)
    if not match:
        return None
    prefix = line[: match.start()].strip()
    if len(prefix) > 140 or re.search(r"[.!?]", prefix):
        return None
    rest = line[match.end() :].strip()
    rest = re.split(
        r"\bpalavras\s*[-–—]?\s*chaves?\b|\babstract\b|\bintrodu",
        rest,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    cleaned = rest.strip(" .:-")
    return cleaned or None


def _split_keywords(blob: str) -> tuple[str, ...]:
    blob = re.sub(r"\s+", " ", blob).strip(" .:-")
    if not blob:
        return ()
    pieces = re.split(r"\s*[,;]\s*", blob)
    keywords = tuple(piece.strip(" .:-") for piece in pieces if piece.strip(" .:-"))
    return keywords[:12]


def _is_resumo_heading(line: str) -> bool:
    if len(line) > 40:
        return False
    cleaned = re.sub(r"^(?:[ivxlcdm]+|\d+)\s+", "", line.strip(), flags=re.IGNORECASE)
    return cleaned.strip(" .:-").casefold() == "resumo"
