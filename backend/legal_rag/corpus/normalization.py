"""Adapters from heterogeneous JSON records to the canonical case schema."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from typing import Any

from legal_rag.corpus.schema import CanonicalCase


class NormalizationError(ValueError):
    """Raised when a source record cannot become a canonical case."""


_ALIASES: dict[str, tuple[str, ...]] = {
    "title": (
        "title",
        "case_title",
        "case_name",
        "name",
        "judgment_title",
    ),
    "case_number": (
        "case_number",
        "case_no",
        "case_num",
        "docket_number",
        "diary_number",
    ),
    "court": ("court", "court_name", "tribunal", "forum"),
    "judgment_date": (
        "judgment_date",
        "decision_date",
        "date_of_judgment",
        "decided_on",
        "date",
    ),
    "source": ("source", "provider", "dataset", "source_name"),
    "source_url": (
        "source_url",
        "url",
        "judgment_url",
        "document_url",
        "link",
    ),
    "raw_text": (
        "raw_text",
        "text",
        "judgment_text",
        "full_text",
        "content",
        "body",
    ),
}

_NESTED_CONTAINER_KEYS = ("metadata", "details", "document", "case", "data")
_KEY_NORMALIZER_RE = re.compile(r"[^a-z0-9]+")
_SCALAR_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_key(value: object) -> str:
    return _KEY_NORMALIZER_RE.sub("", str(value).casefold())


def _containers(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    containers = [record]
    normalized_record = {_normalize_key(key): value for key, value in record.items()}
    for key in _NESTED_CONTAINER_KEYS:
        candidate = normalized_record.get(_normalize_key(key))
        if isinstance(candidate, Mapping):
            containers.append(candidate)
    return containers


def _first_value(
    containers: Sequence[Mapping[str, Any]], aliases: Sequence[str]
) -> Any:
    normalized_aliases = tuple(_normalize_key(alias) for alias in aliases)
    for container in containers:
        values = {_normalize_key(key): value for key, value in container.items()}
        for alias in normalized_aliases:
            value = values.get(alias)
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            if isinstance(value, (Mapping, Sequence)) and not value:
                continue
            return value
    return None


def _scalar(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (Mapping, list, tuple, set)):
        return None
    cleaned = _SCALAR_WHITESPACE_RE.sub(" ", str(value)).strip()
    return cleaned or None


def _raw_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        nested = _first_value([value], ("text", "content", "body", "value"))
        return _raw_text(nested)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        parts = [_raw_text(part) for part in value]
        joined = "\n\n".join(part for part in parts if part and part.strip())
        return joined or None
    return str(value)


def _parse_date(value: Any) -> date | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value

    raw_value = str(value).strip()
    if not raw_value:
        return None

    iso_candidate = raw_value.removesuffix("Z")
    try:
        return datetime.fromisoformat(iso_candidate).date()
    except ValueError:
        pass

    for date_format in (
        "%d-%m-%Y",
        "%d/%m/%Y",
        "%Y/%m/%d",
        "%d.%m.%Y",
        "%d %B %Y",
        "%d %b %Y",
        "%B %d, %Y",
        "%b %d, %Y",
    ):
        try:
            return datetime.strptime(raw_value, date_format).date()
        except ValueError:
            continue
    raise NormalizationError(f"unsupported judgment date: {raw_value!r}")


def normalize_record(
    record: Mapping[str, Any], *, default_source: str | None = None
) -> CanonicalCase:
    """Normalize one heterogeneous JSON mapping into :class:`CanonicalCase`.

    Keys are compared case-insensitively while ignoring separators, which also
    makes camelCase variants work. Common metadata containers are searched after
    the top-level record.
    """

    if not isinstance(record, Mapping):
        raise NormalizationError("record must be a JSON object")

    containers = _containers(record)
    values = {
        field: _first_value(containers, aliases)
        for field, aliases in _ALIASES.items()
    }

    raw_text = _raw_text(values["raw_text"])
    if not raw_text or not raw_text.strip():
        raise NormalizationError("record has no usable judgment text")

    case_number = _scalar(values["case_number"])
    title = _scalar(values["title"]) or case_number or "Untitled case"
    source = _scalar(values["source"]) or _scalar(default_source)

    try:
        return CanonicalCase(
            title=title,
            raw_text=raw_text,
            case_number=case_number,
            court=_scalar(values["court"]),
            judgment_date=_parse_date(values["judgment_date"]),
            source=source,
            source_url=_scalar(values["source_url"]),
        )
    except (TypeError, ValueError) as exc:
        if isinstance(exc, NormalizationError):
            raise
        raise NormalizationError(str(exc)) from exc
