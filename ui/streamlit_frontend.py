from __future__ import annotations

import logging
import os
import uuid
from typing import Any

import httpx
import streamlit as st

# ── Constants ─────────────────────────────────────────────────────────

# API_BASE_URL lets Docker Compose override the default for container-to-container
# networking (e.g. "http://api:8001"). Local dev keeps the default.
_DEFAULT_API_URL: str = os.getenv("API_BASE_URL", "http://localhost:8001")
_REQUEST_TIMEOUT: float = 120.0
_NO_MATCH: str = "NO_MATCH"
_MSG_UNREACHABLE: str = "⚠️ Could not reach the backend."
_MSG_KB_NOT_READY: str = (
    "⚠️ The knowledge base is not loaded yet. "
    "Please ingest documents first using the sidebar."
)

logger = logging.getLogger(__name__)

# ── Page configuration ────────────────────────────────────────────────

st.set_page_config(
    page_title="Agentic RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state initialisation ──────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Active corpus: either the default curated PDFs or a user-uploaded file.
if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = None

if "uploaded_chunks" not in st.session_state:
    st.session_state.uploaded_chunks = 0


# ── API helpers ───────────────────────────────────────────────────────


def _api_base_url() -> str:
    """Return the user-configured backend URL (no trailing slash)."""
    return st.session_state.get("api_url", _DEFAULT_API_URL).rstrip("/")


def _call_api(method: str, path: str, **kwargs: Any) -> httpx.Response | None:
    """Send an HTTP request to the backend and surface connection errors.

    Returns the raw ``httpx.Response`` on success, or ``None`` when the
    request could not be completed (error banner shown automatically).
    """
    url = f"{_api_base_url()}/api{path}"
    try:
        with httpx.Client(timeout=_REQUEST_TIMEOUT) as client:
            return client.request(method, url, **kwargs)
    except httpx.ConnectError:
        logger.warning("Backend unreachable at %s", url)
        st.error(
            f"**Cannot reach the API at `{_api_base_url()}`.**  \n"
            "Make sure the backend is running:  \n"
            "```\nuvicorn src.api.app:app --reload --port 8001\n```"
        )
    except httpx.ReadTimeout:
        logger.warning("Request to %s timed out", url)
        st.error(
            "**Request timed out.**  \n"
            "The backend took too long to respond. Try again or check the server logs."
        )
    except Exception:
        logger.exception("Unexpected HTTP error for %s %s", method, url)
        st.error("**An unexpected error occurred.** Check the console for details.")
    return None


# ── UI component helpers ──────────────────────────────────────────────


def _render_verification_warning(verification: dict[str, Any]) -> None:
    """Display a warning banner when the verifier flags issues."""
    lines: list[str] = [
        "⚠️ **The verifier found potential issues with this answer.**",
    ]
    for claim in verification.get("unsupported_claims", []):
        lines.append(f"- **Unsupported:** {claim}")
    for contradiction in verification.get("contradictions", []):
        lines.append(f"- **Contradiction:** {contradiction}")
    if note := verification.get("additional_details"):
        lines.append(f"**Note:** {note}")
    st.warning("\n\n".join(lines))


def _render_message_badges(meta: dict[str, Any]) -> None:
    """Render relevance and verification badges for a stored message."""
    if meta.get("relevance_status") == _NO_MATCH:
        st.info("ℹ️ No relevant documents were found for this question.")
    verification = meta.get("verification")
    if verification and not verification.get("supported"):
        _render_verification_warning(verification)


def _render_sources(sources: list[dict[str, Any]]) -> None:
    """Render a de-duplicated source list inside an expander."""
    if not sources:
        return
    with st.expander(f"📄 Sources ({len(sources)})"):
        seen: set[str] = set()
        for src in sources:
            key = src.get("source", "")
            if key in seen:
                continue
            seen.add(key)
            page = src.get("page_number")
            page_str = f" (p. {page})" if page else ""
            st.write(f"- **{key}**{page_str}")


def _render_guardrails(guardrails: dict[str, Any]) -> None:
    """Render PII guardrails report inside an expander."""
    input_found = guardrails.get("input_pii_found", False)
    output_found = guardrails.get("output_pii_found", False)
    if not input_found and not output_found:
        return

    with st.expander("🛡️ PII Guardrails"):
        detections = guardrails.get("detections", [])
        pii_types: dict[str, int] = {}
        for d in detections:
            t = d.get("pii_type", "UNKNOWN")
            pii_types[t] = pii_types.get(t, 0) + 1

        if input_found:
            count = guardrails.get("input_redactions", 0)
            st.write(f"**Input query:** {count} PII detection(s) redacted")
        if output_found:
            count = guardrails.get("output_redactions", 0)
            st.write(f"**Draft answer:** {count} PII detection(s) redacted")

        if pii_types:
            type_strs = [f"{count} {ptype}" for ptype, count in pii_types.items()]
            st.write(f"**Types found:** {', '.join(type_strs)}")


def _render_audit_trail(session_id: str) -> None:
    """Render the audit trail for this session inside an expander."""
    if not session_id:
        return

    with st.expander("📋 Audit Trail"):
        resp = _call_api("GET", f"/audit/{session_id}")
        if resp is None:
            st.write("Could not fetch audit trail.")
            return
        if resp.status_code == 404:
            st.write("No audit events recorded for this session.")
            return
        if resp.status_code != 200:
            st.write(f"Error fetching audit trail: {resp.status_code}")
            return

        events = resp.json()
        if not events:
            st.write("No audit events recorded.")
            return

        for event in events:
            ts = event.get("timestamp", "")
            if "T" in ts:
                ts = ts.split("T")[1][:8]  # HH:MM:SS
            node = event.get("node_name", "")
            etype = event.get("event_type", "")
            st.write(f"`{ts}` **{node}** → {etype}")

        st.caption(f"{len(events)} event(s) total")


def _append_message(
    role: str,
    content: str,
    meta: dict[str, Any] | None = None,
) -> None:
    """Append a message to session state."""
    entry: dict[str, Any] = {"role": role, "content": content}
    if meta:
        entry["meta"] = meta
    st.session_state.messages.append(entry)


# ── Sidebar ───────────────────────────────────────────────────────────


def _render_sidebar() -> None:
    """Build the sidebar with settings, health check, and ingestion."""
    with st.sidebar:
        st.title("⚙️ Settings")

        st.text_input(
            "Backend API URL",
            value=_DEFAULT_API_URL,
            key="api_url",
            help="Base URL of the FastAPI backend.",
        )

        st.divider()

        # Health probe
        st.subheader("System Status")
        if st.button("🔄 Check Health", use_container_width=True):
            _handle_health_check()

        st.divider()

        # Upload your own PDF — scoped to this session only, isolated from
        # other users and from the default curated corpus.
        st.subheader("📤 Upload Your Own PDF")
        st.caption(
            "Drop a PDF here and the chatbot will answer questions about "
            "*your* document instead of the default demo corpus. "
            "Your upload stays private to this session."
        )
        uploaded = st.file_uploader(
            "Choose a PDF (max 20 MB)",
            type=["pdf"],
            accept_multiple_files=False,
            label_visibility="collapsed",
        )
        if uploaded is not None and uploaded.name != st.session_state.uploaded_filename:
            _handle_upload(uploaded)

        if st.session_state.uploaded_filename:
            st.success(
                f"✅ Now querying **{st.session_state.uploaded_filename}** "
                f"({st.session_state.uploaded_chunks} chunks)"
            )
            if st.button("↩️ Back to default corpus", use_container_width=True):
                st.session_state.uploaded_filename = None
                st.session_state.uploaded_chunks = 0
                st.session_state.session_id = str(uuid.uuid4())
                st.session_state.messages = []
                st.rerun()

        st.divider()

        # Clear conversation
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.uploaded_filename = None
            st.session_state.uploaded_chunks = 0
            st.rerun()


def _handle_health_check() -> None:
    """Call ``/health`` and display the result."""
    resp = _call_api("GET", "/health")
    if resp is None:
        return
    if resp.status_code != 200:
        st.error(f"Unexpected status: {resp.status_code}")
        return

    data = resp.json()
    if data.get("knowledge_base_ready"):
        st.success(
            f"✅ **Online** — {data['vector_store_type'].title()} "
            f"({data.get('documents_indexed', '?')} chunks)"
        )
    else:
        st.warning(
            "⚠️ **API running** but knowledge base is empty.  \n"
            "Upload PDFs to `data/raw/` and click **Ingest Documents**."
        )


def _handle_upload(uploaded_file: Any) -> None:
    """POST the uploaded PDF to /api/upload and pin the session to its namespace."""
    # Rotate the session id so the new upload gets a fresh namespace and the
    # previous conversation context doesn't leak into audit-trail lookups.
    st.session_state.session_id = str(uuid.uuid4())

    with st.spinner(f"Processing {uploaded_file.name} …"):
        resp = _call_api(
            "POST",
            "/upload",
            files={
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    "application/pdf",
                ),
            },
            data={"session_id": st.session_state.session_id},
        )

    if resp is None:
        return
    if resp.status_code != 200:
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        st.error(f"Upload failed: {detail}")
        return

    data = resp.json()
    st.session_state.uploaded_filename = data["filename"]
    st.session_state.uploaded_chunks = data["total_chunks"]
    st.session_state.messages = []  # fresh chat for the new doc
    st.rerun()


# ── Chat area ─────────────────────────────────────────────────────────


def _render_chat_history() -> None:
    """Re-render all past messages (including badges) from session state."""
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            _render_message_badges(msg.get("meta", {}))


def _handle_chat_response(resp: httpx.Response | None) -> None:
    """Process the backend response inside an already-opened assistant bubble."""
    if resp is None:
        st.markdown(_MSG_UNREACHABLE)
        _append_message("assistant", _MSG_UNREACHABLE)
        return

    if resp.status_code == 503:
        st.warning(_MSG_KB_NOT_READY)
        _append_message("assistant", _MSG_KB_NOT_READY)
        return

    if resp.status_code != 200:
        detail = _safe_detail(resp)
        answer = f"⚠️ Error from backend: {detail}"
        st.error(answer)
        _append_message("assistant", answer)
        return

    data: dict[str, Any] = resp.json()
    answer: str = data["answer"]
    st.markdown(answer)

    meta: dict[str, Any] = {
        "relevance_status": data.get("relevance_status"),
        "verification": data.get("verification"),
        "iterations": data.get("iterations", 0),
        "sources": data.get("sources", []),
        "guardrails": data.get("guardrails"),
    }

    _render_message_badges(meta)
    _render_sources(data.get("sources", []))

    guardrails = data.get("guardrails")
    if guardrails:
        _render_guardrails(guardrails)

    _render_audit_trail(data.get("session_id", ""))

    iterations = data.get("iterations", 0)
    if iterations and iterations > 1:
        st.caption(f"🔄 Answer refined through {iterations} iteration(s).")

    _append_message("assistant", answer, meta)


def _safe_detail(resp: httpx.Response) -> str:
    """Extract a human-readable error detail from a failed response."""
    try:
        return resp.json().get("detail", resp.text)
    except Exception:
        return resp.text or f"HTTP {resp.status_code}"


# ── Main ──────────────────────────────────────────────────────────────


def main() -> None:
    """Application entry-point: sidebar, history, and chat input loop."""
    _render_sidebar()

    st.title("🤖 Agentic RAG Chatbot")
    st.caption(
        "Ask questions about the documents in the knowledge base. "
        "The system retrieves relevant chunks, drafts an answer, and "
        "verifies it for accuracy."
    )

    if st.session_state.uploaded_filename:
        st.info(
            f"📄 **Querying your uploaded document:** "
            f"`{st.session_state.uploaded_filename}`"
        )
    else:
        st.info(
            "📚 **Querying the default corpus:** DeepSeek Technical Report, "
            "NVIDIA Annual Report 2025, Google Environmental Report 2024, "
            "Impact Report 2025, NASDAQ NVDA 2024, OpenAI Nonprofit Commission. "
            "Upload your own PDF in the sidebar to query something else."
        )

    _render_chat_history()

    if prompt := st.chat_input("Ask a question …"):
        _append_message("user", prompt)
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking …"):
                resp = _call_api(
                    "POST",
                    "/chat",
                    json={
                        "question": prompt,
                        "session_id": st.session_state.session_id,
                    },
                )
            _handle_chat_response(resp)


if __name__ == "__main__":
    main()
else:
    main()
