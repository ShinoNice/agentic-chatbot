"""Tests for the Pydantic Settings loader.

These tests verify that YAML → Pydantic mapping is stable against the
shape we rely on elsewhere (magic-number extraction, CORS lock-down,
per-endpoint limits). They do not hit the network or require real secrets.
"""

from __future__ import annotations

from src.core.config_loader import (
    AgentSettings,
    AppSettings,
    DoclingSettings,
    UploadSettings,
    settings,
)


def test_settings_exposes_core_sections():
    """The loaded settings singleton carries every section routes/agents depend on."""
    assert settings.llm.primary_llm
    assert settings.rag.top_k > 0
    assert isinstance(settings.app.cors_origins, list)
    assert settings.upload.max_size_mib > 0
    assert settings.agent.max_verifier_context_chars > 0
    assert settings.docling.timeout_seconds > 0
    assert settings.docling.max_pages > 0


def test_upload_max_size_bytes_is_mib_times_1_mib():
    """max_size_bytes is a derived property — guard against drift."""
    up = UploadSettings(max_size_mib=25)
    assert up.max_size_bytes == 25 * 1024 * 1024


def test_cors_origins_default_is_localhost_only():
    """If no config overrides it, the default CORS allow-list is dev-only.

    Prevents regressions that reintroduce a wildcard default.
    """
    app = AppSettings()
    assert "*" not in app.cors_origins
    assert any("localhost" in origin for origin in app.cors_origins)


def test_docling_settings_have_safe_defaults():
    """Docling knobs have non-zero defaults (timeout + page cap)."""
    d = DoclingSettings()
    assert d.timeout_seconds >= 60
    assert d.max_pages >= 10


def test_agent_settings_default_context_cap_is_reasonable():
    """Verifier/relevance checker char cap stays under typical LLM window.

    12k chars ≈ 3k tokens — well under 4o-mini's 128k window but large
    enough to pass meaningful context.
    """
    a = AgentSettings()
    assert 4_000 <= a.max_verifier_context_chars <= 32_000


def test_loaded_cors_origins_contains_production_fqdn():
    """The committed settings.yaml includes the deployed Azure UI origin.

    Guards against accidental reverts to "*" or removal of the prod origin.
    """
    origins = settings.app.cors_origins
    assert "*" not in origins, "CORS must not be wildcard in committed config"
    assert any("azurecontainerapps.io" in o for o in origins), (
        "Production Streamlit UI FQDN must be in the allow-list"
    )


def test_claim_pipeline_settings_loaded():
    # `enabled` is the only field that flips during rollout; assert the
    # other knobs match the YAML defaults.
    assert isinstance(settings.claim_pipeline.enabled, bool)
    assert settings.claim_pipeline.max_repair_rounds == 1
    assert settings.claim_pipeline.drafter_max_claims == 12
    assert settings.claim_pipeline.verify_concurrency == 5
    assert settings.claim_pipeline.total_budget_seconds == 60
