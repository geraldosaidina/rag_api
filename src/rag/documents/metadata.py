"""Deterministic PFC cover and front-matter metadata extraction."""

from __future__ import annotations

import re

from rag.documents.front_matter import extract_abstract, extract_keywords
from rag.documents.models import ExtractedPfcMetadata

_MONTHS = (
    "janeiro|fevereiro|março|marco|abril|maio|junho|julho|agosto|"
    "setembro|outubro|novembro|dezembro"
)
_INSTITUTION = re.compile(
    r"(instituto\s+superior\s+de\s+transportes\s+e\s+comunica[cç][oõ]es)",
    re.IGNORECASE,
)
_PROJECTO = re.compile(r"projecto\s+final", re.IGNORECASE)
_YEAR = re.compile(rf"\b(?:{_MONTHS})\s*,?\s*(?:de\s+)?((?:19|20)\d{{2}})\b", re.IGNORECASE)
_COURSE = re.compile(
    rf"((?:curso\s+de\s+)?licenciatura\b.*?)(?=\s+supervisora?\b|\s+departamento\b|\s+dtic\b|"
    rf"\s+eng(?:[ªº.]|\b)|\s+dr[ª.]|\s+prof\.?\b|\s+msc\b|\s+(?:{_MONTHS})\b|$)",
    re.IGNORECASE,
)
_SUPERVISOR_LABEL = re.compile(r"supervisora?\s*:\s*(.*)", re.IGNORECASE)
_ACK_SUPERVISOR = re.compile(
    r"supervisor(?:a)?(?:\s+docente)?\s*,?\s+(?:o\s+|a\s+)?"
    r"((?:eng|dr|prof|msc|ms|dra)\.?ª?\.?\s*[^,.\n]{2,80})",
    re.IGNORECASE,
)
_NAME_TAIL = re.compile(
    r"("
    r"[A-ZÁÉÍÓÚÂÊÔÃÕÇ][a-záéíóúâêôãõç'’.-]+"
    r"(?:(?:\s+(?:da|de|do|dos|das|e)\s+|\s+)[A-ZÁÉÍÓÚÂÊÔÃÕÇ][a-záéíóúâêôãõç'’.-]+){1,6}"
    r")\s*$"
)


def extract_pfc_metadata(pages: list[str]) -> ExtractedPfcMetadata:
    """Parse title, author, year, and optional front matter from PDF page texts."""
    cover: dict[str, str | int | None] = {
        "title": None,
        "author": None,
        "year": None,
        "course": None,
        "institution": None,
        "department": None,
        "supervisor": None,
    }
    warnings: list[str] = []
    for page in pages[:3]:
        parsed = _parse_cover(page)
        for key, value in parsed.items():
            if value and not cover.get(key):
                cover[key] = value

    supervisor = cover.get("supervisor")
    if not supervisor:
        supervisor = _acknowledgement_supervisor(pages[:15])
        if supervisor:
            warnings.append("supervisor_from_acknowledgement")

    abstract = extract_abstract(pages)
    keywords = extract_keywords(pages)
    if abstract and not keywords:
        warnings.append("keywords_missing")
    if not abstract and cover.get("title"):
        warnings.append("abstract_missing")

    language = "pt" if cover.get("title") or abstract else None
    author = cover.get("author")
    return ExtractedPfcMetadata(
        title=_clean(cover.get("title")),
        authors=(str(author),) if author else (),
        year=cover.get("year") if isinstance(cover.get("year"), int) else None,
        course=_clean(cover.get("course")),
        institution=_clean(cover.get("institution")),
        department=_clean(cover.get("department")),
        supervisor=_clean(supervisor),
        abstract=abstract,
        keywords=keywords,
        language=language,
        warnings=tuple(warnings),
    )


def _parse_cover(text: str) -> dict[str, str | int | None]:
    lines = _explode_projecto(_cover_lines(text))
    projecto_at = next((i for i, line in enumerate(lines) if _PROJECTO.search(line)), None)
    tail = lines[projecto_at:] if projecto_at is not None else []
    content = _preamble_content(lines[:projecto_at] if projecto_at is not None else [])
    title, author = _title_and_author(content)
    return {
        "title": title,
        "author": author,
        "year": _year(text),
        "course": _course(tail),
        "institution": _institution(text),
        "department": _department(tail),
        "supervisor": _labelled_supervisor(tail),
    }


def _cover_lines(text: str) -> list[str]:
    normalized = text.replace("\u00a0", " ").replace("\r", "\n")
    if normalized.count("\n") < 3:
        normalized = re.sub(r"[ \t]{3,}", "\n", normalized)
    lines: list[str] = []
    for raw in normalized.splitlines():
        line = re.sub(r"\s+", " ", raw).strip()
        if line:
            lines.append(line)
    return lines


def _explode_projecto(lines: list[str]) -> list[str]:
    exploded: list[str] = []
    for line in lines:
        match = _PROJECTO.search(line)
        if match and match.start() > 0:
            head = line[: match.start()].strip()
            tail = line[match.start() :].strip()
            if head:
                exploded.append(head)
            if tail:
                exploded.append(tail)
        else:
            exploded.append(line)
    return exploded


def _preamble_content(lines: list[str]) -> list[str]:
    content: list[str] = []
    for line in lines:
        match = _INSTITUTION.search(line)
        if match and match.start() == 0:
            rest = line[match.end() :].strip(" .:-")
            if rest:
                content.append(rest)
            continue
        if match and line.strip().casefold() == match.group(1).casefold():
            continue
        content.append(line)
    return content


def _title_and_author(content: list[str]) -> tuple[str | None, str | None]:
    if not content:
        return None, None
    if len(content) == 1:
        title, author = _split_title_author(content[0])
        return title, author
    author = content[-1].strip(" .:-\"'")
    title = " ".join(content[:-1]).strip(" .:-\"'")
    if len(author) > 120:
        split_title, split_author = _split_title_author(author)
        if split_author:
            title = " ".join(part for part in (title, split_title) if part)
            author = split_author
    return title or None, author or None


def _split_title_author(line: str) -> tuple[str | None, str | None]:
    match = _NAME_TAIL.search(line.strip())
    if not match:
        return _clean(line), None
    author = match.group(1).strip()
    title = line[: match.start()].strip(" .:-\"'")
    if len(title) < 8:
        return _clean(line), None
    return title, author


def _institution(text: str) -> str | None:
    match = _INSTITUTION.search(re.sub(r"\s+", " ", text))
    if not match:
        return None
    return re.sub(r"\s+", " ", match.group(1)).strip()


def _year(text: str) -> int | None:
    match = _YEAR.search(re.sub(r"\s+", " ", text))
    if not match:
        return None
    return int(match.group(1))


def _course(lines: list[str]) -> str | None:
    for line in lines:
        if not re.search(r"licenciatura", line, flags=re.IGNORECASE):
            continue
        match = _COURSE.search(line)
        if match:
            return match.group(1).strip(" .:-")
    return None


def _department(lines: list[str]) -> str | None:
    month = re.compile(rf"\b(?:{_MONTHS})\b", re.IGNORECASE)
    for line in lines:
        if re.fullmatch(r"dtic", line.strip(), flags=re.IGNORECASE):
            return line.strip()
        if re.match(r"departamento\b", line, flags=re.IGNORECASE):
            return month.split(line, maxsplit=1)[0].strip(" .:-")
    return None


def _labelled_supervisor(lines: list[str]) -> str | None:
    for index, line in enumerate(lines):
        match = _SUPERVISOR_LABEL.match(line.strip())
        if not match:
            continue
        name = match.group(1).strip(" .:-")
        if not name and index + 1 < len(lines):
            name = lines[index + 1].strip(" .:-")
        name = re.split(r"\bdepartamento\b", name, maxsplit=1, flags=re.IGNORECASE)[0]
        name = re.split(rf"\b(?:{_MONTHS})\b", name, maxsplit=1, flags=re.IGNORECASE)[0]
        cleaned = name.strip(" .:-")
        return cleaned or None
    return None


def _acknowledgement_supervisor(pages: list[str]) -> str | None:
    for page in pages:
        match = _ACK_SUPERVISOR.search(page.replace("\n", " "))
        if match:
            return match.group(1).strip(" .:-")
    return None


def _clean(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = re.sub(r"\s+", " ", value).strip(" .:-\"'")
    return cleaned or None
