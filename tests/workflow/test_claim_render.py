from src.schemas.claim_schemas import Citation, Claim, ClaimStatus
from src.workflow.claim_pipeline.render import render_final_answer, FALLBACK_NO_CLAIMS


def test_render_empty_returns_fallback():
    assert render_final_answer([]) == FALLBACK_NO_CLAIMS


def test_render_emits_one_claim_tag_per_claim_with_status():
    claims = [
        Claim(id="c1", text="A is B.", status=ClaimStatus.VERIFIED,
              citations=[Citation(chunk_id="ck1", quote="A is B")]),
        Claim(id="c2", text="C might be D.", status=ClaimStatus.UNSUPPORTED, citations=[]),
    ]
    out = render_final_answer(claims)
    assert '<claim id="c1" status="verified" data-cite="ck1">A is B.</claim>' in out
    assert '<claim id="c2" status="unsupported" data-cite="">C might be D.</claim>' in out


def test_render_escapes_html_in_claim_text():
    claims = [
        Claim(id="c1", text="<script>alert(1)</script>", status=ClaimStatus.VERIFIED, citations=[]),
    ]
    out = render_final_answer(claims)
    assert "<script>" not in out
    assert "&lt;script&gt;" in out
