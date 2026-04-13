"""Tests for PII pattern detection and redaction.

All tests are pure-function regex tests — zero mocks, zero API calls.
"""

import asyncio

from src.mcp.guardrails.patterns import (
    _validate_luhn,
    _validate_nif,
    scan_text,
)
from src.mcp.guardrails.server import redact_text, scan_pii


# ── NIF (Portuguese tax ID) ─────────────────────────────────────────


class TestNIF:
    def test_scan_nif_valid(self):
        # 123456789 passes mod-11: weights [9,8,7,6,5,4,3,2]
        # sum = 1*9+2*8+3*7+4*6+5*5+6*4+7*3+8*2 = 9+16+21+24+25+24+21+16 = 156
        # 156%11 = 2, check = 11-2 = 9 ✓
        text = "O NIF do cliente é 123456789 registado."
        result = scan_text(text)
        nif_hits = [d for d in result if d.pii_type == "NIF"]
        assert len(nif_hits) == 1
        assert nif_hits[0].value == "123456789"

    def test_scan_nif_with_pt_prefix(self):
        text = "NIF: PT123456789"
        result = scan_text(text)
        nif_hits = [d for d in result if d.pii_type == "NIF"]
        assert len(nif_hits) == 1
        assert "123456789" in nif_hits[0].value

    def test_scan_nif_invalid_checksum_rejected(self):
        # 123456780 — check digit 0 != expected 9
        text = "NIF: 123456780"
        result = scan_text(text)
        nif_hits = [d for d in result if d.pii_type == "NIF"]
        assert len(nif_hits) == 0

    def test_validate_nif_helper(self):
        assert _validate_nif("123456789") is True
        assert _validate_nif("000000000") is False  # starts with 0
        assert _validate_nif("123456780") is False  # bad check digit


# ── Portuguese Phone ────────────────────────────────────────────────


class TestPhonePT:
    def test_scan_pt_phone_mobile(self):
        text = "Contacte-nos: 912 345 678"
        result = scan_text(text)
        phone_hits = [d for d in result if d.pii_type == "PHONE_PT"]
        assert len(phone_hits) == 1

    def test_scan_pt_phone_landline(self):
        text = "Telefone: 213 456 789"
        result = scan_text(text)
        phone_hits = [d for d in result if d.pii_type == "PHONE_PT"]
        assert len(phone_hits) == 1

    def test_scan_pt_phone_with_country_code(self):
        text = "Ligar para +351 912 345 678"
        result = scan_text(text)
        # May match as PHONE_PT or PHONE_INTL — either is valid
        phone_hits = [d for d in result if d.pii_type in ("PHONE_PT", "PHONE_INTL")]
        assert len(phone_hits) >= 1


# ── NISS (Social Security) ──────────────────────────────────────────


class TestNISS:
    def test_scan_niss_with_keyword(self):
        text = "O NISS do trabalhador é 12345678901."
        result = scan_text(text)
        niss_hits = [d for d in result if d.pii_type == "NISS"]
        assert len(niss_hits) == 1
        assert niss_hits[0].value == "12345678901"

    def test_scan_niss_without_keyword_not_detected(self):
        # Without a keyword nearby, 11 digits alone should not trigger NISS
        text = "O número é 12345678901."
        result = scan_text(text)
        niss_hits = [d for d in result if d.pii_type == "NISS"]
        assert len(niss_hits) == 0


# ── Cartão de Cidadão ───────────────────────────────────────────────


class TestCartaoCidadao:
    def test_scan_cartao_cidadao(self):
        text = "CC: 12345678 9 ZZ1"
        result = scan_text(text)
        cc_hits = [d for d in result if d.pii_type == "CARTAO_CIDADAO"]
        assert len(cc_hits) == 1


# ── IBAN ────────────────────────────────────────────────────────────


class TestIBAN:
    def test_scan_iban_pt(self):
        text = "IBAN: PT50 0002 0123 1234 5678 9015 4"
        result = scan_text(text)
        iban_hits = [d for d in result if d.pii_type == "IBAN"]
        assert len(iban_hits) == 1
        assert iban_hits[0].value.startswith("PT50")

    def test_scan_iban_generic_eu(self):
        text = "Account: DE89 3704 0044 0532 0130 00"
        result = scan_text(text)
        iban_hits = [d for d in result if d.pii_type == "IBAN"]
        assert len(iban_hits) == 1
        assert iban_hits[0].value.startswith("DE89")


# ── Código Postal ───────────────────────────────────────────────────


class TestCodigoPostal:
    def test_scan_codigo_postal(self):
        text = "Morada: Rua X, 1000-001 Lisboa"
        result = scan_text(text)
        cp_hits = [d for d in result if d.pii_type == "CODIGO_POSTAL"]
        assert len(cp_hits) == 1
        assert cp_hits[0].value == "1000-001"


# ── Email ───────────────────────────────────────────────────────────


class TestEmail:
    def test_scan_email(self):
        text = "Envie para joao.silva@empresa.pt por favor."
        result = scan_text(text)
        email_hits = [d for d in result if d.pii_type == "EMAIL"]
        assert len(email_hits) == 1
        assert email_hits[0].value == "joao.silva@empresa.pt"


# ── Credit Card ─────────────────────────────────────────────────────


class TestCreditCard:
    def test_scan_credit_card_valid_luhn(self):
        # Visa test number (passes Luhn)
        text = "Card: 4111 1111 1111 1111"
        result = scan_text(text)
        cc_hits = [d for d in result if d.pii_type == "CREDIT_CARD"]
        assert len(cc_hits) == 1

    def test_scan_credit_card_invalid_luhn_rejected(self):
        text = "Card: 4111 1111 1111 1112"
        result = scan_text(text)
        cc_hits = [d for d in result if d.pii_type == "CREDIT_CARD"]
        assert len(cc_hits) == 0

    def test_validate_luhn_helper(self):
        assert _validate_luhn("4111111111111111") is True
        assert _validate_luhn("4111111111111112") is False
        assert _validate_luhn("12345") is False  # too short


# ── IPv4 ────────────────────────────────────────────────────────────


class TestIP:
    def test_scan_ip_valid(self):
        text = "Server at 192.168.1.100"
        result = scan_text(text)
        ip_hits = [d for d in result if d.pii_type == "IP"]
        assert len(ip_hits) == 1
        assert ip_hits[0].value == "192.168.1.100"

    def test_scan_ip_out_of_range_rejected(self):
        text = "Address: 999.999.999.999"
        result = scan_text(text)
        ip_hits = [d for d in result if d.pii_type == "IP"]
        assert len(ip_hits) == 0


# ── Date of Birth ───────────────────────────────────────────────────


class TestDOB:
    def test_scan_dob_with_keyword_pt(self):
        text = "Data de nascimento: 15/01/1990"
        result = scan_text(text)
        dob_hits = [d for d in result if d.pii_type == "DOB"]
        assert len(dob_hits) == 1

    def test_scan_dob_with_keyword_en(self):
        text = "Date of birth: 01-15-1990"
        result = scan_text(text)
        dob_hits = [d for d in result if d.pii_type == "DOB"]
        assert len(dob_hits) == 1

    def test_scan_dob_without_keyword_not_detected(self):
        text = "The date 15/01/1990 is important."
        result = scan_text(text)
        dob_hits = [d for d in result if d.pii_type == "DOB"]
        assert len(dob_hits) == 0


# ── US SSN ──────────────────────────────────────────────────────────


class TestSSN:
    def test_scan_ssn(self):
        text = "SSN: 123-45-6789"
        result = scan_text(text)
        ssn_hits = [d for d in result if d.pii_type == "SSN"]
        assert len(ssn_hits) == 1
        assert ssn_hits[0].value == "123-45-6789"


# ── International Phone ─────────────────────────────────────────────


class TestPhoneIntl:
    def test_scan_intl_phone(self):
        text = "Call +44 20 7946 0958"
        result = scan_text(text)
        phone_hits = [d for d in result if d.pii_type == "PHONE_INTL"]
        assert len(phone_hits) == 1


# ── Edge Cases ──────────────────────────────────────────────────────


class TestEdgeCases:
    def test_scan_no_pii_returns_empty(self):
        text = "This is a perfectly normal sentence with no sensitive data."
        result = scan_text(text)
        assert len(result) == 0

    def test_scan_empty_text(self):
        result = scan_text("")
        assert result == []

    def test_scan_multiple_pii_types(self):
        text = "Email: joao@test.pt, NIF: 123456789, IP: 10.0.0.1"
        result = scan_text(text)
        types_found = {d.pii_type for d in result}
        assert "EMAIL" in types_found
        assert "NIF" in types_found
        assert "IP" in types_found

    def test_detections_sorted_by_start(self):
        text = "Email: a@b.pt then NIF: 123456789"
        result = scan_text(text)
        if len(result) >= 2:
            assert result[0].start <= result[1].start


# ── Redaction (FastMCP server tools) ────────────────────────────────





class TestScanPiiTool:
    def test_scan_pii_returns_result(self):
        result = asyncio.run(scan_pii("Email: test@example.com"))
        assert result["has_pii"] is True
        assert result["scanned_length"] > 0
        assert len(result["detections"]) == 1

    def test_scan_pii_no_pii(self):
        result = asyncio.run(scan_pii("Nothing sensitive here."))
        assert result["has_pii"] is False
        assert result["detections"] == []


class TestRedaction:
    def test_redact_mask_strategy(self):
        text = "Email: joao@test.pt"
        result = asyncio.run(redact_text(text, strategy="mask"))
        assert "[REDACTED_EMAIL]" in result["redacted_text"]
        assert "joao@test.pt" not in result["redacted_text"]
        assert result["redactions_applied"] == 1
        assert result["strategy"] == "mask"

    def test_redact_hash_strategy(self):
        text = "Email: joao@test.pt"
        result = asyncio.run(redact_text(text, strategy="hash"))
        assert "[EMAIL:" in result["redacted_text"]
        assert "joao@test.pt" not in result["redacted_text"]
        assert result["strategy"] == "hash"

    def test_redact_remove_strategy(self):
        text = "Email: joao@test.pt ok"
        result = asyncio.run(redact_text(text, strategy="remove"))
        assert "joao@test.pt" not in result["redacted_text"]
        assert "Email:" in result["redacted_text"]
        assert result["strategy"] == "remove"

    def test_redact_preserves_non_pii_text(self):
        text = "Hello world, no PII here."
        result = asyncio.run(redact_text(text))
        assert result["redacted_text"] == text
        assert result["redactions_applied"] == 0

    def test_redact_multiple_pii_types(self):
        text = "Email: a@b.pt, NIF: 123456789, IP: 10.0.0.1"
        result = asyncio.run(redact_text(text, strategy="mask"))
        assert "a@b.pt" not in result["redacted_text"]
        assert "123456789" not in result["redacted_text"]
        assert "10.0.0.1" not in result["redacted_text"]
        assert result["redactions_applied"] == 3

    def test_redact_no_pii_returns_original(self):
        text = "Safe text with nothing to redact."
        result = asyncio.run(redact_text(text))
        assert result["redacted_text"] == text
        assert result["redactions_applied"] == 0
        assert result["original_length"] == len(text)
