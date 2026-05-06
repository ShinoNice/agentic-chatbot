# Frontend Migration: Streamlit → Angular

This document captures what changed when the demo UI moved from
`ui/streamlit_frontend.py` (Streamlit) to `frontend/` (Angular 21 SPA), and
what would have to change in the deployment ("lift-and-shift") to take the
new frontend live alongside, or in place of, the existing Streamlit
container.

---

## TL;DR

| | Before (Streamlit) | After (Angular) |
| --- | --- | --- |
| Stack | Python + Streamlit | Angular 21 standalone + signals + Tailwind v4 |
| Where it lives | `ui/streamlit_frontend.py` | `frontend/` |
| Backend talk | `POST /api/chat` (single blocking JSON) | `POST /api/chat/stream` (SSE, per-node events) |
| State | `st.session_state` (server, per-process) | Browser signals + `localStorage` |
| Multi-session history | None — refresh wiped chat | Up to 50 sessions persisted client-side |
| Sources / verifier / PII / audit | Stacked `st.expander` blocks under each answer | Inline ribbon + slide-out inspector with tabs |
| Live progress | Spinner ("Thinking…") | Pipeline stepper: `Retrieve → Draft → Verify` glowing on the active node |
| Markdown answers | Streamlit defaults | `marked` + `DOMPurify`, with `[1]/[2]` citations rewired into hover-popover chips |
| Theme | Hard-coded dark | Dark / light toggle, persisted |
| Build artefact | None (runs as Python process) | Static SPA in `frontend/dist/angular-app/browser/` |

The FastAPI backend is unchanged in behaviour but gained one additive
endpoint (`POST /api/chat/stream`) and one new dependency (`sse-starlette`).
The legacy `/chat` endpoint stays for non-streaming consumers, including
the existing Streamlit UI.

---

## What actually changed

### Backend (additive only)

* `POST /api/chat/stream` — Server-Sent Events. Same request body as `/chat`.
  Emits `step` events as each LangGraph node completes, then a `result`
  event carrying the same `ChatResponse` shape `/chat` returns, then `done`.
  Errors surface as an `error` event.
* `RAGOrchestrator.astream_run` — wraps `app.astream(..., stream_mode="updates")`
  and translates each node's update into a `step` payload.
* `_build_chat_response` — extracted from the `/chat` handler so both
  endpoints share the response-building code.
* `sse-starlette` added to `pyproject.toml`.

The Streamlit UI keeps using `/chat` and is unaffected. Both UIs can run
side-by-side against the same backend.

### Frontend

Angular 21 standalone components, signals throughout, Tailwind v4.
Layout is three regions: collapsible left sidebar, main chat, slide-out
right inspector.

* **Live pipeline stepper** above the composer — the signature visual.
  Driven by a `streamingStep` signal in `SessionStore` that the chat
  component pokes as `step` events arrive over SSE.
* **Inline citation chips** — the markdown renderer rewrites `[n]`
  references into superscript `<sup>` chips that pop up the source
  snippet on click.
* **Verification / guardrails / no-match ribbon** — inline above the
  answer, surfaces problems immediately instead of hiding them in
  expanders.
* **Per-message inspector** — Sources / Verify / PII / Audit tabs.
  Replaces the four stacked Streamlit expanders.
* **`localStorage` session history** — sessions, messages, active id,
  API base URL, theme, and uploaded-corpus state all persist across
  refresh. Cap of 50 sessions, oldest dropped.
* **Drag-and-drop PDF upload** in the sidebar (POST `/api/upload`),
  same wire format as before.
* **API URL editor + health dot** — green/amber/red indicator, click
  the refresh icon to re-probe `/api/health`.
* **Dark / light theme toggle**, persisted.

### Bug fixes shipped during the rebuild

These were caught while bringing the new UI up:

* **`postcss.config.mjs` was silently ignored** by Angular's build
  pipeline, which only auto-discovers `postcss.config.json` /
  `.postcssrc.json`. Result: Tailwind v4 never ran and the SPA shipped
  with zero utility classes — pure unstyled HTML in the browser.
  Fixed by renaming to `.postcssrc.json`.
* **SSE parser only split on `\n\n`**, but `sse-starlette` writes
  `\r\n\r\n` between events. Frames were never detected, so the
  `result` event never reached the chat component and assistant
  messages were stuck on the loading dots forever. Parser now tries
  `\r\n\r\n` / `\n\n` / `\r\r` in priority order.
* **`http://localhost:4200` was missing from `app.cors_origins`** in
  `config/settings.yaml` — the Angular dev server's preflight was
  rejected with `400`. Added.

---

## Lift-and-shift to production

The current Azure Container Apps deployment ships exactly one frontend:
`ca-chatbot-ui`, a Streamlit container with public ingress on `:8501`.
Defined in `infra/main.bicep` (look for `uiAppName`, `chatbot-ui` image,
`streamlit run ui/streamlit_frontend.py`) and `docker-compose.yml`
(service `ui`).

Two viable shapes for taking the Angular app live:

### Option A — replace the Streamlit container with an Angular static-site container (recommended)

Smallest surface area, smallest blast radius, no infra rename.

1. **Add a frontend Dockerfile** (`frontend/Dockerfile`):
   * Stage 1: `node:22-alpine`, run `npm ci && npm run build`.
   * Stage 2: `nginx:alpine`, copy `dist/angular-app/browser/` to
     `/usr/share/nginx/html`. Add a tiny `nginx.conf` with
     `try_files $uri $uri/ /index.html;` so client-side routing works.
   * Listen on `:80` (or whatever ACA expects).

2. **Build and push to ACR** as `chatbot-ui:<tag>` (same image name
   the Bicep already references — the swap stays invisible to IaC).

3. **Update `docker-compose.yml`** so the `ui` service builds from
   `frontend/` and exposes Nginx's port (e.g. `4200:80`). Drop the
   `command: streamlit ...` override and the `API_BASE_URL` env var
   (the browser hits the API directly, not through compose DNS).

4. **Update `infra/main.bicep`**:
   * `targetPort: 80` (was `8501`) on the `uiApp` ingress.
   * Remove `command: ['streamlit']` and `args: [...]` from the
     container — Nginx is the entrypoint.
   * The `API_INTERNAL_URL` env var is no longer used (the browser
     can't resolve internal ACA DNS) — remove it.

5. **Bake the public API URL into the build, OR resolve at runtime.**
   In dev the Angular app reads it from a sidebar input + `localStorage`,
   defaulting to `http://localhost:8001`. In prod two options:
   * **Build-time:** swap the default in `frontend/src/app/core/api.service.ts`
     to the public API FQDN (`https://ca-chatbot-api.<region>.azurecontainerapps.io`)
     before `npm run build`. Cheapest.
   * **Runtime:** ship a `assets/config.json` that's served by Nginx and
     fetched at app boot. Lets one image run against multiple
     environments. Strictly better but a bit more wiring.

6. **Make the API publicly reachable.** Today `ca-chatbot-api` has
   `external: false` (internal-only). The browser is outside the ACA
   environment, so it needs `external: true` *or* a reverse proxy on
   the UI container that forwards `/api/*` to the API. Pick one:
   * Flip the API ingress to public and add the UI's public FQDN to
     `app.cors_origins` (already a Bicep param — `corsExtraOrigins`).
   * Or keep the API internal and add an Nginx `location /api/` block
     on the UI container that `proxy_pass`es to `http://ca-chatbot-api`.
     Avoids CORS entirely and keeps the API off the public internet.
     Mildly preferable for a demo deployment.

7. **Delete the Streamlit code path** when you're confident:
   `ui/streamlit_frontend.py` and the `streamlit` dependency in
   `pyproject.toml` (~150 MB out of the runtime image). The `/chat`
   endpoint can stay or be removed — it has no other consumer.

### Option B — keep both UIs running side-by-side

Useful if you want to A/B them, or keep Streamlit as a fallback while
the Angular app gets traffic.

1. Add a *new* container app (`ca-chatbot-ui-angular` or rename the
   existing one) with its own ingress hostname.
2. Same Nginx + static-build container as Option A.
3. Streamlit container stays as-is.
4. CORS allow-list grows by one FQDN.
5. Roughly doubles the UI infra cost (one extra replica), so only do
   this temporarily.

### What does NOT change

* The API container, its image, its scaling, its KV-backed secrets, the
  managed identity, the Log Analytics workspace, the ACR — none of
  these care which UI is in front.
* Audit / guardrails / vector-store data on the persistent volume —
  unchanged.
* Existing CI workflows that build & push `chatbot-api:<tag>` — unchanged.
* The `chatbot-ui` ACR image *name* — keep it the same so Bicep stays
  stable. Only the contents of the image (Streamlit → Nginx + static
  bundle) change.

### Rough effort

Option A end-to-end is in the order of half a day:
~1 hour for the Dockerfile + Nginx config, ~1 hour for the Bicep tweaks
and a manual deploy, ~1 hour to wire the API URL handling, the rest is
testing and rollback verification. Option B adds another ~1 hour for
the duplicate container app and DNS.

---

## Local dev quick reference

```
# Backend
uv run uvicorn src.api.app:app --reload --port 8001

# Streamlit (legacy, still works)
streamlit run ui/streamlit_frontend.py

# Angular (new)
cd frontend && npm install && npm start    # http://localhost:4200
```

`http://localhost:4200`, `:8501` and `:8001` are all in
`app.cors_origins`, so any of them can talk to the API directly.
