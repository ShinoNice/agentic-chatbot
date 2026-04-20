"""Regex-based PII pattern detection for PT/EU and international formats.

All detection is deterministic (regex + heuristic validation).
Zero API cost, fully testable without mocks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.schemas.agent_schemas import PIIDetection


@dataclass
class PIIPattern:
    """Definition of a PII pattern to scan for."""

    name: str
    regex: re.Pattern[str]
    confidence: float = 1.0
    keyword_gate: list[str] | None = field(default=None)
    validator: str | None = None  # name of validation function


# ── Validation helpers ───────────────────────────────────────────────


def _validate_nif(digits: str) -> bool:
    """Validate Portuguese NIF using mod-11 check digit."""
    clean = digits.replace(" ", "")
    if len(clean) != 9 or not clean.isdigit():
        return False
    # First digit must be 1-9
    if clean[0] == "0":
        return False
    weights = [9, 8, 7, 6, 5, 4, 3, 2]
    total = sum(int(clean[i]) * weights[i] for i in range(8))
    remainder = total % 11
    check = 0 if remainder < 2 else 11 - remainder
    return int(clean[8]) == check


def _validate_luhn(digits: str) -> bool:
    """Validate credit card number using the Luhn algorithm."""
    clean = re.sub(r"[\s\-]", "", digits)
    if not clean.isdigit() or len(clean) < 13 or len(clean) > 19:
        return False
    total = 0
    for i, ch in enumerate(reversed(clean)):
        n = int(ch)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def _validate_ip_range(match: str) -> bool:
    """Validate that each IPv4 octet is 0-255."""
    parts = match.split(".")
    if len(parts) != 4:
        return False
    return all(p.isdigit() and 0 <= int(p) <= 255 for p in parts)


# ── Pattern definitions ─────────────────────────────────────────────

# Portuguese NIF: 9 digits, optionally prefixed with "PT"
_NIF_RE = re.compile(r"(?<!\d)(?:PT\s?)?(\d{9})(?!\d)")

# Portuguese phone: +351 or 9xx/2xx start, 9 digits total
_PHONE_PT_RE = re.compile(
    r"(?:\+351[\s\-]?)?(?:(?:9[1-9]\d|2[1-9]\d)[\s\-]?\d{3}[\s\-]?\d{3})(?!\d)"
)

# NISS (Social Security): 11 digits, keyword-gated
_NISS_RE = re.compile(r"(?<!\d)(\d{11})(?!\d)")

# Cartão de Cidadão: 8 digits + check digit + 2 letters + 1 digit
_CC_RE = re.compile(r"(?<!\w)\d{8}\s?\d\s?[A-Z]{2}\d(?!\w)")

# IBAN: 2-letter country + 2 check digits + up to 30 alphanumeric
_IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}[\s]?(?:\d{4}[\s]?){2,7}\d{1,4}\b")

# Código Postal: XXXX-XXX
_CODIGO_POSTAL_RE = re.compile(r"(?<!\d)\d{4}-\d{3}(?!\d)")

# Email: simplified RFC 5322
_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")

# Credit card: 13-19 digits with optional separators
_CREDIT_CARD_RE = re.compile(r"(?<!\d)(\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{1,7})(?!\d)")

# IPv4 address
_IP_RE = re.compile(r"(?<!\d)(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})(?!\d)")

# Date of birth: keyword-gated date patterns
_DOB_RE = re.compile(r"(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})")
_DOB_KEYWORDS = [
    "dob",
    "born",
    "birth",
    "nascimento",
    "nascido",
    "nascida",
    "data de nascimento",
    "date of birth",
]

# US SSN: XXX-XX-XXXX
_SSN_RE = re.compile(r"(?<!\d)(\d{3}-\d{2}-\d{4})(?!\d)")

# International phone: + country code then 7-14 digits
_PHONE_INTL_RE = re.compile(r"\+\d{1,3}[\s\-]?\d[\d\s\-]{6,13}\d(?!\d)")

# NISS keywords for gating
_NISS_KEYWORDS = [
    "niss",
    "segurança social",
    "seguranca social",
    "social security",
    "número de segurança",
]


# ── All patterns ────────────────────────────────────────────────────

PII_PATTERNS: list[PIIPattern] = [
    PIIPattern(name="NIF", regex=_NIF_RE, validator="nif"),
    PIIPattern(name="PHONE_PT", regex=_PHONE_PT_RE),
    PIIPattern(name="NISS", regex=_NISS_RE, keyword_gate=_NISS_KEYWORDS),
    PIIPattern(name="CARTAO_CIDADAO", regex=_CC_RE),
    PIIPattern(name="IBAN", regex=_IBAN_RE),
    PIIPattern(name="CODIGO_POSTAL", regex=_CODIGO_POSTAL_RE),
    PIIPattern(name="EMAIL", regex=_EMAIL_RE),
    PIIPattern(name="CREDIT_CARD", regex=_CREDIT_CARD_RE, validator="luhn"),
    PIIPattern(name="IP", regex=_IP_RE, validator="ip"),
    PIIPattern(name="DOB", regex=_DOB_RE, keyword_gate=_DOB_KEYWORDS, confidence=0.9),
    PIIPattern(name="SSN", regex=_SSN_RE),
    PIIPattern(name="PHONE_INTL", regex=_PHONE_INTL_RE),
]

# Validators by name
_VALIDATORS = {
    "nif": _validate_nif,
    "luhn": _validate_luhn,
    "ip": _validate_ip_range,
}


# ── Public API ──────────────────────────────────────────────────────


def _has_keyword(text: str, keywords: list[str], match_start: int) -> bool:
    """Check if any keyword appears within 80 chars before the match."""
    window_start = max(0, match_start - 80)
    window = text[window_start:match_start].lower()
    return any(kw in window for kw in keywords)


def _resolve_overlaps(detections: list[PIIDetection]) -> list[PIIDetection]:
    """Remove overlapping detections, keeping the longer/more-specific match."""
    if not detections:
        return []
    sorted_dets = sorted(detections, key=lambda d: (d.start, -(d.end - d.start)))
    result: list[PIIDetection] = [sorted_dets[0]]
    for det in sorted_dets[1:]:
        prev = result[-1]
        if det.start < prev.end:
            # Overlap: keep the longer one
            if (det.end - det.start) > (prev.end - prev.start):
                result[-1] = det
        else:
            result.append(det)
    return result


def scan_text(text: str) -> list[PIIDetection]:
    """Scan text for all known PII patterns.

    Returns a list of PIIDetection objects sorted by start offset,
    with overlaps resolved (longer match wins).
    """
    if not text:
        return []

    detections: list[PIIDetection] = []

    for pattern in PII_PATTERNS:
        for match in pattern.regex.finditer(text):
            matched_text = match.group(0)
            start = match.start()

            # Keyword gating: skip if required keyword not found nearby
            if pattern.keyword_gate:
                if not _has_keyword(text, pattern.keyword_gate, start):
                    continue

            # Validation: skip if validator rejects the match
            if pattern.validator:
                validator_fn = _VALIDATORS.get(pattern.validator)
                if validator_fn:
                    # For NIF, validate the digit group, not the "PT" prefix
                    validate_text = matched_text
                    if pattern.validator == "nif":
                        validate_text = re.sub(r"^PT\s?", "", matched_text)
                    if not validator_fn(validate_text):
                        continue

            detections.append(
                PIIDetection(
                    pii_type=pattern.name,
                    value=matched_text,
                    start=start,
                    end=match.end(),
                    confidence=pattern.confidence,
                )
            )

    return _resolve_overlaps(detections)
