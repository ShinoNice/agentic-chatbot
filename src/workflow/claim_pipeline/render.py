from html import escape

from src.schemas.claim_schemas import Claim


FALLBACK_NO_CLAIMS = (
    "I'm sorry, but the available context did not support a confident answer "
    "to that question."
)


def render_final_answer(claims: list[Claim]) -> str:
    """Render the final answer as a string of `<claim>` tags.

    The Angular frontend parses these tags into per-claim badge components.
    HTML in claim text is escaped to prevent injection through the LLM output.
    """
    if not claims:
        return FALLBACK_NO_CLAIMS

    parts = []
    for c in claims:
        cite_ids = ",".join(cite.chunk_id for cite in c.citations)
        parts.append(
            f'<claim id="{escape(c.id)}" status="{c.status.value}" '
            f'data-cite="{escape(cite_ids)}">{escape(c.text)}</claim>'
        )
    return " ".join(parts)
