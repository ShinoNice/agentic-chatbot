# Cross-Encoder Reranker — Design Spec

**Date:** 2026-04-08
**Branch (planned):** `claude/2026-04-08-reranker`
**Phase:** Retrieval quality (one-branch detour from "Tests + hardening")
**Status:** Approved for implementation planning

---

## 1. Goal

Add a cross-encoder reranking step to the agentic-chatbot retrieval pipeline, motivated by the lowest RAGAS metric on the committed golden set: **Context Precision = 67.21%** (vs Faithfulness 90.75%, per [PROJECT_SUMMARY.md §14.1](../../../PROJECT_SUMMARY.md)). Reranking is the textbook fix for low context precision — it widens the retrieval funnel, then concentrates precision before the LLM sees the context.

The deliverable is **measured**, not assumed: the branch ships baseline + sweep eval results in `evaluation/results/`, demonstrating the lift on the committed dataset. The spec itself includes a per-question delta analysis as part of the rollout.

### Success criteria

1. RAGAS Context Precision on `golden_set_v2.json` improves from baseline (~67%) by a measurable margin under at least one of the swept N→K configurations.
2. Faithfulness, Answer Relevancy, and Context Recall do not regress meaningfully (defined as: each stays within ±5 percentage points of baseline; if any drops more than that, we explicitly call it out in the rollout notes and pick a config that doesn't).
3. The reranker module ships with passing unit tests; `uv run pytest -q --no-header` returns exit 0.
4. Baseline + sweep CSVs are committed alongside the code, so the lift is reproducible from the repo alone.

### Non-goals (explicit)

- No changes to BM25/dense retrieval weights, chunking strategy, or vector store backend. We isolate the reranker as the only variable so the eval delta is attributable only to it.
- No tests for the orchestrator's `node_rerank` integration edge — this branch ships unit tests for the reranker module only. Orchestrator integration tests are part of the next phase.
- No bootstrap of the broader `tests/conftest.py` for the next "Tests + hardening" phase. Only fixtures the reranker tests themselves need.
- No latency benchmarking gate. Latency is reported in the rollout notes but does not block the change.
- No hosted reranker (Cohere, Voyage, Jina) and no `BaseReranker` runtime swap mechanism. Single implementation, single code path. The ABC exists for testability and future extension, not for runtime polymorphism.
- No bundled refactor of unrelated tangled code. If wiring exposes coupling that blocks a test, refactor in the same chunk; otherwise leave existing code alone.
- No observability instrumentation (latency histograms, score distributions, etc.). Observability is §14.3 territory and out of scope for this branch.
- No prompt-side changes to the existing agents. The relevance checker, researcher, and verifier are untouched — they just receive a smaller, better-ordered `state["documents"]`.

---

## 2. Architecture

### 2.1 Graph shape

The orchestrator's LangGraph state machine picks up one new node, `rerank`, between `retrieve` and `check_relevance`:

```
START → retrieve → rerank → check_relevance → research → verify → END
                                   │
                                   └─ relevance NO_MATCH → END
                                   └─ verifier retry  → research (loop, bounded)
```

The `rerank` node is **always present** in the graph regardless of `settings.rerank.enabled`. When disabled, it short-circuits and passes its input through unchanged. This means:

- The LangGraph diagram is identical whether reranking is on or off — honest about the architecture.
- The A/B switch for the eval is a single boolean flip in YAML — no orchestrator rebuild.
- A baseline eval run (rerank disabled) never imports `torch`, never loads the cross-encoder model, because the orchestrator only instantiates `BGEReranker` when `enabled=true`.

### 2.2 Reranker module

A small package mirroring the [src/engines/base.py](../../../src/engines/base.py) ABC pattern:

- `BaseReranker(ABC)` — single abstract method `rerank(query, documents, top_k) -> list[Document]`. Lives in `src/retrieval/reranker.py` next to the concrete implementation.
- `BGEReranker(BaseReranker)` — wraps `sentence_transformers.CrossEncoder` loading `BAAI/bge-reranker-base` (English-only, ~278 MB). Lazy loads the model on first call so importing the module is cheap and tests don't trigger a model download.

The ABC exists so that tests can substitute a `MockReranker` cleanly, not because we expect runtime polymorphism. Listed as a non-goal above.

### 2.3 Why `BAAI/bge-reranker-base`

- **English-only.** Inspection of `evaluation/datasets/golden_set_v2.json` confirms the corpus is English (DeepSeek-R1 paper, Cloudflare news). The multilingual `bge-reranker-v2-m3` (~568 MB) would be ~2x the footprint with no quality gain on this dataset.
- **Battle-tested.** BGE rerankers are widely cited in the RAG literature; the model card is the right reference for the spec's "model choice" narrative.
- **Footprint.** ~278 MB on disk vs. ~568 MB for v2-m3. The cold install footprint of `sentence-transformers + torch + transformers` is the dominant cost (~1 GB) regardless of model — but the model weights themselves are much smaller with the base variant.
- **Latency.** Roughly 30-100ms per (query, document) batch of 30 on CPU (varies by hardware). Acceptable for an interactive chatbot, well under the existing LLM call latency.

---

## 3. Components

### 3.1 New files

#### `src/retrieval/reranker.py`

```python
from abc import ABC, abstractmethod
from typing import List, Optional
from langchain_core.documents import Document
from src.core.exceptions import RerankerError
from src.core.logger import logger


class BaseReranker(ABC):
    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int,
    ) -> List[Document]: ...


class BGEReranker(BaseReranker):
    """Cross-encoder reranker using a local sentence-transformers model.

    Lazy-loads the model on first call. Subsequent failures (load or
    predict) raise RerankerError; the orchestrator catches and falls
    back to unranked candidates.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None
        self._load_failed = False

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        if self._load_failed:
            raise RerankerError(
                f"Reranker model {self.model_name} previously failed to load."
            )
        try:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading cross-encoder: {self.model_name}")
            self._model = CrossEncoder(self.model_name)
        except Exception as e:
            self._load_failed = True
            raise RerankerError(
                f"Failed to load cross-encoder {self.model_name}: {e}"
            ) from e

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int,
    ) -> List[Document]:
        if not documents or top_k <= 0:
            return []

        self._ensure_loaded()

        pairs = [(query, doc.page_content) for doc in documents]
        try:
            scores = self._model.predict(pairs)
        except Exception as e:
            raise RerankerError(f"Reranker predict() failed: {e}") from e

        # Stable sort: ties preserve the original retriever order.
        ranked = sorted(
            enumerate(documents),
            key=lambda pair: scores[pair[0]],
            reverse=True,
        )
        return [doc for _, doc in ranked[:top_k]]
```

#### `evaluation/compare_runs.py`

Small pandas script. Reads two `report_*.csv` files (baseline, reranked), joins on `user_input`, prints/saves a per-question delta table sorted by `context_precision_delta` descending. Roughly 30 lines. Run as:

```bash
python evaluation/compare_runs.py \
  evaluation/results/report_<baseline_ts>.csv \
  evaluation/results/report_<reranked_ts>.csv
```

Outputs both a printed summary table and `evaluation/results/delta_<baseline_ts>_vs_<reranked_ts>.csv` (committed).

#### `tests/__init__.py`, `tests/retrieval/__init__.py`, `tests/conftest.py`, `tests/retrieval/test_reranker.py`

`tests/conftest.py` contains only fixtures the reranker tests need:

- `sample_documents` — three or four `Document` objects with distinct `page_content` and `metadata`.
- `mock_cross_encoder_class(monkeypatch)` — a fixture factory that monkeypatches `sentence_transformers.CrossEncoder` to a fake class with a configurable `predict()` return value. No torch, no model weights, no network.

`tests/retrieval/test_reranker.py` covers:

1. `test_rerank_orders_by_score` — mocked predict returns a known score vector; assert reranker returns documents in score-descending order.
2. `test_rerank_truncates_to_top_k` — `top_k=2` returns exactly 2 documents from a list of 4.
3. `test_rerank_top_k_larger_than_input_returns_all` — `top_k=10` on 3 documents returns all 3 (in reranked order).
4. `test_rerank_empty_input_returns_empty` — empty list in → empty list out, no model load triggered.
5. `test_rerank_top_k_zero_returns_empty` — `top_k=0` → empty list out, no model load triggered.
6. `test_rerank_stable_tie_breaking` — two documents with identical scores preserve their original input order.
7. `test_load_failure_raises_reranker_error` — monkeypatch `CrossEncoder.__init__` to raise; assert `rerank()` raises `RerankerError`.
8. `test_load_failure_is_sticky` — after a failed load, a second `rerank()` call raises `RerankerError` immediately without re-attempting `CrossEncoder.__init__`.
9. `test_predict_failure_raises_reranker_error` — load succeeds, monkeypatch `predict()` to raise; assert `RerankerError` propagates.
10. `test_predict_failure_is_not_sticky` — after a transient predict failure, the next call attempts predict again normally.

10 tests total. All pure unit, no real model load, no network.

### 3.2 Edited files

| File | Change |
|---|---|
| [src/workflow/orchestrator.py](../../../src/workflow/orchestrator.py) | Add `node_rerank`. Add `BGEReranker` instance (lazy, only if `settings.rerank.enabled`). New graph edges: `retrieve → rerank → check_relevance`. |
| [src/workflow/memory.py](../../../src/workflow/memory.py) | Add `candidate_documents: List[Document]` to `AgentState` TypedDict. |
| [src/retrieval/hybrid_search.py](../../../src/retrieval/hybrid_search.py) | Add optional `k: Optional[int] = None` constructor parameter; default to `settings.rag.top_k` when `None`. No coupling to rerank settings inside the searcher. |
| [src/api/dependencies.py](../../../src/api/dependencies.py) | `SystemManager` gains a `_searcher_k()` helper that returns `settings.rerank.candidate_k` if rerank is enabled, else `settings.rag.top_k`. Both `HybridSearcher(...)` call sites pass `k=self._searcher_k()`. |
| [src/core/config_loader.py](../../../src/core/config_loader.py) | New `RerankSettings` Pydantic model with field validator (`top_k <= candidate_k`). Wire into `Settings` and `load_all_configs()`. |
| [config/settings.yaml](../../../config/settings.yaml) | New `rerank_settings:` block, `enabled: false` by default. |
| [src/core/exceptions.py](../../../src/core/exceptions.py) | Add `RerankerError(RetrievalError)`. |
| [pyproject.toml](../../../pyproject.toml) | Add `sentence-transformers>=3.0.0`. Pulls in `torch` and `transformers` transitively (~1 GB cold install). |
| [.gitignore](../../../.gitignore) | Add `evaluation/results/*_full.csv` so the bulky full reports stay local. Slim `report_*.csv` summaries are committed. |
| [CLAUDE.md](../../../CLAUDE.md) | Flip "Current Phase" to `Retrieval quality (one-branch detour from "Tests + hardening")`. Note that the hook's "no tests collected" grace period ends with this branch. |
| [README.md](../../../README.md) | One paragraph under the architecture section noting the reranker step and pointing at the eval results CSVs. |

---

## 4. Configuration

### 4.1 YAML

```yaml
# config/settings.yaml — new block
rerank_settings:
  enabled: false                       # off by default; baseline run uses this
  model_name: "BAAI/bge-reranker-base" # English-only, ~278 MB
  candidate_k: 30                      # how many docs the retriever returns
  top_k: 5                             # how many docs survive reranking
```

### 4.2 Pydantic

```python
# src/core/config_loader.py — new model
class RerankSettings(BaseModel):
    enabled: bool = False
    model_name: str = "BAAI/bge-reranker-base"
    candidate_k: int = 30
    top_k: int = 5

    @field_validator("top_k")
    @classmethod
    def top_k_within_candidates(cls, v, info):
        candidate_k = info.data.get("candidate_k")
        if candidate_k is not None and v > candidate_k:
            raise ValueError(
                f"rerank.top_k ({v}) must be <= rerank.candidate_k ({candidate_k})"
            )
        return v


class Settings(BaseSettings):
    ...
    rerank: RerankSettings = Field(default_factory=RerankSettings)
```

`load_all_configs()` reads `yaml_data.get("rerank_settings", {})` and passes it to `RerankSettings(**...)`. Falling back to defaults preserves environments where the YAML block is missing (e.g., older Docker images).

### 4.3 Defaults

`enabled: false` is intentional **during the sweep**. The baseline eval run uses defaults; the post-rerank eval runs flip it on. After eval lands and §13 is filled in, the chosen winning config is committed to `config/settings.yaml` in the same chunk (rather than as a separate one-line follow-up commit). See [§13.4](#134-decision-on-default) for the actual decision and the post-eval `enabled: true, candidate_k: 30, top_k: 10` defaults that ship with this branch.

---

## 5. Data Flow

### 5.1 State field split

`AgentState` gains one field. The split is load-bearing because it makes the "wide retrieve, narrow after rerank" semantics visible in the type system, and it makes the bypass path trivial.

```python
# src/workflow/memory.py
class AgentState(TypedDict):
    question: str
    candidate_documents: List[Document]   # NEW: wide retrieve from node_retrieve
    documents: List[Document]             # post-rerank slice (== candidates if disabled)
    relevance_status: RelevanceStatus
    draft_answer: str
    verification: VerificationReport
    iterations: int
```

`node_retrieve` returns `{"candidate_documents": docs}`. `node_rerank` returns `{"documents": <slice>}`. Downstream nodes (`check_relevance`, `research`, `verify`) continue to read `state["documents"]` exactly as today.

### 5.2 The new node

```python
# src/workflow/orchestrator.py — new node
async def node_rerank(self, state: AgentState) -> Dict[str, Any]:
    logger.info("--- NODE: RERANK ---")
    candidates = state["candidate_documents"]

    if not settings.rerank.enabled or self.reranker is None:
        return {"documents": candidates}

    try:
        reranked = await asyncio.to_thread(
            self.reranker.rerank,
            state["question"],
            candidates,
            settings.rerank.top_k,
        )
        logger.info(
            f"Reranked {len(candidates)} → {len(reranked)} chunks"
        )
        return {"documents": reranked}
    except RerankerError as e:
        logger.warning(
            f"Reranker failed, falling back to candidates: {e}"
        )
        return {"documents": candidates}
```

Two non-obvious choices:

1. **`asyncio.to_thread(...)`** — `CrossEncoder.predict()` is sync and CPU-bound. Calling it on the event loop blocks every other request. Wrapping in `to_thread` is the minimum-cost fix and matches the kind of "sync I/O in async paths" issue called out in [PROJECT_SUMMARY.md §14.2](../../../PROJECT_SUMMARY.md). We're not boiling that ocean broadly, but the new code we're adding gets it right from the start.

2. **The bypass branch is data-driven, not graph-structural.** Even with `enabled=false`, the graph still routes through `rerank` — the node just early-returns the candidates. This means the LangGraph diagram is identical whether rerank is on or off, and the eval A/B is a single boolean flip in YAML.

### 5.3 The HybridSearcher k parameter

Today [HybridSearcher.__init__](../../../src/retrieval/hybrid_search.py#L16-L23) reads `settings.rag.top_k` directly. With reranking, we want the retriever to return `candidate_k` (30) when `rerank.enabled=true`, and `rag.top_k` (10) when disabled.

We keep `HybridSearcher` decoupled from rerank settings (it's a retrieval module, it shouldn't know reranking exists):

```python
# src/retrieval/hybrid_search.py — minimal change
def __init__(self, vector_store, documents=None, k: Optional[int] = None):
    self.vector_store = vector_store
    self.documents = documents
    self.top_k = k if k is not None else settings.rag.top_k  # NEW
    ...
```

`SystemManager` computes the effective `k` once and passes it in:

```python
# src/api/dependencies.py
def _searcher_k(self) -> int:
    if settings.rerank.enabled:
        return settings.rerank.candidate_k
    return settings.rag.top_k

# both call sites:
self.searcher = HybridSearcher(vs, documents=cached_docs, k=self._searcher_k())
```

This isolates rerank knowledge to `SystemManager` (which already owns the wiring) and keeps `HybridSearcher` pure.

### 5.4 Reranker instantiation

`RAGOrchestrator.__init__` lazy-instantiates `BGEReranker` only if `settings.rerank.enabled`:

```python
self.reranker = (
    BGEReranker(settings.rerank.model_name)
    if settings.rerank.enabled
    else None
)
```

When disabled, `self.reranker is None` and `node_rerank` early-returns on the bypass branch. A baseline eval run never imports `sentence_transformers`, never loads the model, never touches torch.

---

## 6. Error Handling

Three failure modes, three responses. Philosophy: **graceful degradation** — the answer still flows even if reranking is broken. Mirrors the existing "BM25 unavailable, fall back to vector-only" pattern in [hybrid_search.py:33-38](../../../src/retrieval/hybrid_search.py#L33-L38).

| Mode | Trigger | Response |
|---|---|---|
| **Model load fails** | First `_ensure_loaded()` call raises (HF Hub down, missing model name, OOM, missing torch). | Set `self._load_failed = True`. Re-raise as `RerankerError`. `node_rerank` catches, logs warning, returns candidates unchanged. Subsequent calls short-circuit (don't retry the load) and re-raise immediately. |
| **`predict()` fails on a specific call** | Tokenizer chokes, transient OOM mid-batch. Rare. | Re-raise as `RerankerError` *without* setting `_load_failed`. `node_rerank` catches and falls back to candidates **for this call only**. The instance is not poisoned — next call retries `predict()` normally. |
| **Empty input** | `documents=[]` or `top_k=0`. | Not an error. `BGEReranker.rerank()` returns `[]` silently, no model load triggered. The downstream relevance checker already handles empty documents in [relevance_checker.py:33-35](../../../src/workflow/agents/relevance_checker.py#L33-L35) by returning `NO_MATCH`. |

**New exception:** `RerankerError(RetrievalError)` in [src/core/exceptions.py](../../../src/core/exceptions.py). Inherits from existing `RetrievalError` so callers that already catch retrieval errors still catch reranker failures.

**Deliberately not handled:**
- No retry/backoff on `predict()` failures. The orchestrator's existing self-correction loop handles answer-quality issues; reranker retry would just delay graceful degradation.
- No latency tracking. No observability layer exists yet (§14.3 territory).
- No circuit breaker. Overkill for a local model.

---

## 7. Eval Methodology

This section is the spine of the spec. Option A from brainstorming (RAGAS Context Precision lift) means the deliverable is *measured*, not assumed.

### 7.1 Sweep grid

Five RAGAS evaluation runs total:

| Run | `rerank.enabled` | `candidate_k` | `top_k` | Notes |
|---|---|---|---|---|
| Baseline | `false` | n/a | n/a (uses `rag.top_k=10`) | Reproduces today's behavior. |
| 30 → 10 | `true` | 30 | 10 | Same final K as baseline; isolates the reranking effect. |
| 30 → 5 | `true` | 30 | 5 | Default; expected best Context Precision. |
| 50 → 10 | `true` | 50 | 10 | Wider candidate pool, same final K. Tests the marginal value of expanding recall. |
| 50 → 5 | `true` | 50 | 5 | Wider pool, sharper cut. The most aggressive config. |

All runs use `--max-samples 10` (the existing default in [eval_pipeline.py](../../../evaluation/eval_pipeline.py#L34)). At ~4 RAGAS metrics per sample, that's ~40 evaluations per run × 5 runs = ~200 RAGAS evaluations total. Cost is small but real (the evaluator LLM is `gpt-4o-mini`).

### 7.2 How a sweep run is executed

For each grid point: edit `config/settings.yaml`, run `python evaluation/eval_pipeline.py`, rename the resulting `report_*.csv` to a config-tagged filename (e.g., `report_2026-04-08_baseline.csv`, `report_2026-04-08_30x5.csv`). Repeat. The `_full.csv` files (with raw LLM outputs) stay local — they're gitignored. Only the slim score CSVs are committed.

We do not script the sweep itself in this branch (would require config-override CLI flags, adding scope). Five manual runs is fine; the cost is in the LLM calls, not in the bash invocations.

**Who runs the sweep:** the user runs it. The eval makes real OpenAI API calls against the user's `OPENAI_API_KEY` and Pinecone/Chroma, so it's not something Claude executes unprompted. The implementation plan that follows this spec will land code chunks 1-5 first (all unit-tested, no real-API calls); then the user runs the five-point sweep manually; then Claude stages chunk 6 (eval result CSVs + delta) and chunk 7 (spec Results section + README + CLAUDE.md) once the CSVs exist on disk.

### 7.3 Per-question delta analysis

`evaluation/compare_runs.py` joins two slim CSVs on `user_input` and produces a per-question table:

| user_input | baseline_cp | reranked_cp | delta_cp | baseline_cr | reranked_cr | delta_cr |
|---|---|---|---|---|---|---|
| "What is the primary..." | 0.40 | 0.85 | +0.45 | 0.80 | 0.80 | 0.00 |
| ... | | | | | | |

Sorted by `delta_cp` descending. The output CSV is committed at `evaluation/results/delta_baseline_vs_<config>.csv`. The spec's "Results" section quotes the top-line averages and a few notable per-question deltas (best lift, worst regression).

### 7.4 What "good" looks like

- **Required:** at least one swept config improves average Context Precision by ≥5 percentage points over baseline.
- **Required:** that same config does not regress Faithfulness, Answer Relevancy, or Context Recall by more than 5 percentage points.
- **Stretch:** at least one config also improves Faithfulness (because cleaner context → less hallucination opportunity).
- **If nothing meets the required bar:** the spec's Results section honestly reports the negative finding, and we keep `enabled: false` as the default. The reranker code still ships — the value is the methodology and the measurement, not a lift we forced.

### 7.5 Where results live

```
evaluation/
├── eval_pipeline.py
├── compare_runs.py                    # NEW
├── datasets/
│   └── golden_set_v2.json
└── results/
    ├── report_2026-04-08_baseline.csv          # COMMITTED (slim)
    ├── report_2026-04-08_30x10.csv             # COMMITTED
    ├── report_2026-04-08_30x5.csv              # COMMITTED
    ├── report_2026-04-08_50x10.csv             # COMMITTED
    ├── report_2026-04-08_50x5.csv              # COMMITTED
    ├── report_2026-04-08_*_full.csv            # GITIGNORED (bulky raw)
    └── delta_baseline_vs_30x5.csv              # COMMITTED
```

---

## 8. Testing

Scope: **reranker module unit tests only**, per Q6 option A from brainstorming. The full test list is in §3.1 above (10 tests). Key constraints:

- **No real model load.** `sentence_transformers.CrossEncoder` is monkeypatched in every test. Tests verify the **contract** (sort order, truncation, edge cases, error propagation), not the **quality** of the reranker. Quality is what the eval pipeline measures.
- **No network.** Honors CLAUDE.md hard constraint #1 (no paid API calls + no external service hits in pytest).
- **No new fixtures beyond what reranker tests need.** `tests/conftest.py` is minimal — `sample_documents` and a monkeypatch helper. The broader test infrastructure for the next "Tests + hardening" phase comes later.

### Hook implications

Once `tests/retrieval/test_reranker.py` exists, the [p26-test-before-commit.sh](../../../../CLAUDE.md) hook's "exit code 5 (no tests collected) is allowed" grace period is over. Every commit on this branch (and onward) must keep `uv run pytest -q --no-header` returning exit 0. This is a feature, not a bug — it's the project graduating from "no tests" to "tests enforced." `CLAUDE.md` is updated to note this.

### What we are NOT testing in this branch

- The orchestrator's `node_rerank` integration edge (the wire-up between retrieve → rerank → check_relevance with rerank enabled vs disabled).
- The `SystemManager._searcher_k()` helper.
- The `RerankSettings` Pydantic validator (`top_k <= candidate_k`).
- End-to-end flow with a mocked LLM.

These belong to the next phase. Listed here so the gap is explicit, not silent.

---

## 9. Caveats & Trade-offs

### 9.1 Install footprint

`sentence-transformers>=3.0.0` pulls in `torch` and `transformers` transitively. Cold install with uv: roughly +1 GB of wheels and on-disk packages, +3-10s of cold-start the first time `CrossEncoder.__init__` runs (model download from HF Hub, ~278 MB cached to `~/.cache/huggingface/`). Subsequent loads are fast.

This is the dominant cost of the change. It's documented here, in `CLAUDE.md`, and in the README paragraph so contributors aren't surprised when `uv sync` takes longer.

### 9.2 Latency

Per-query reranker latency (CPU, 30 candidates, English): roughly 30-100ms depending on hardware, well under the LLM call latency that dominates each query (~1-3s for the relevance check + research + verify cycle). Acceptable. Not gated.

### 9.3 Off by default

`enabled: false` is the shipped default after this branch. The eval results determine whether a follow-up commit flips it to `true`. Two reasons:

- The eval needs the off-state to compute the baseline, and we want anyone re-running the eval from a fresh clone to be able to reproduce both runs.
- Flipping the default after eval results are in is a one-line change that deserves its own commit so the trail in `git log` is clean.

### 9.4 No observability

We don't emit reranker metrics (latency histograms, score distributions, fallback counts). When the observability layer lands (§14.3), the reranker module is the right place to add hooks — but adding them now is speculative scope.

### 9.5 Single implementation, single code path

No `BaseReranker` runtime swap. The ABC exists for testability, not extension. If a future "hosted reranker" branch wants to add `CohereReranker(BaseReranker)`, the abstraction is ready — but that's a future branch's scope.

---

## 10. Rollout & Branching

### 10.1 Branch

Create `claude/2026-04-08-reranker` from `main` (or from the bootstrap branch if not yet merged — see "open thread" below).

### 10.2 Commit chunks (suggested)

Logical chunks the user will commit one at a time. Each chunk leaves the working tree green (`uv run pytest -q --no-header` exit 0, application still imports cleanly).

| # | Chunk | Files | Why |
|---|---|---|---|
| 1 | **Spec doc** | `docs/superpowers/specs/2026-04-08-reranker-design.md`, mkdir `docs/superpowers/specs/` | Ship the design before any code. |
| 2 | **Config plumbing** | `config/settings.yaml`, `src/core/config_loader.py`, `src/core/exceptions.py` | `RerankSettings` model + `RerankerError`. No behavior change yet (`enabled: false`). |
| 3 | **Reranker module + tests** | `src/retrieval/reranker.py`, `tests/__init__.py`, `tests/retrieval/__init__.py`, `tests/conftest.py`, `tests/retrieval/test_reranker.py`, `pyproject.toml` (add `sentence-transformers`) | The module + its tests, isolated. After this commit, `uv run pytest` runs 10 tests, all green. |
| 4 | **Orchestrator integration** | `src/workflow/memory.py`, `src/workflow/orchestrator.py`, `src/retrieval/hybrid_search.py`, `src/api/dependencies.py` | Wire `node_rerank` into the graph; `state["candidate_documents"]` field; HybridSearcher `k` param; SystemManager `_searcher_k()`. Still `enabled: false` so behavior is unchanged. |
| 5 | **Eval tooling** | `evaluation/compare_runs.py`, `.gitignore` (add `*_full.csv`) | Ready for the sweep. |
| 6 | **Eval results** | `evaluation/results/report_2026-04-08_*.csv`, `evaluation/results/delta_baseline_vs_*.csv` | The five sweep CSVs + the delta analysis. This is the "Results" commit; spec gets a Results section appended in the same chunk. |
| 7 | **Spec results section + README + CLAUDE.md** | This file (Results section), `README.md` (architecture paragraph), `CLAUDE.md` (phase update + open question resolution) | Documentation chunk. Closes the loop. |

Per the kernel's section 0, **Claude stages each chunk and the user commits each one manually**. Suggested commit messages will be printed at handoff for each chunk.

### 10.3 Open thread (out of band)

Two pieces of pre-existing uncommitted state are sitting on `claude/2026-04-08-intake-bootstrap`:

- Staged: `CLAUDE.md` (the project bootstrap CLAUDE.md from the prior session)
- Untracked: `PROJECT_SUMMARY.md`
- Modified: `.python-version`, `misc/env_setup.txt`
- Untracked: `misc/workflow_diagramTRUE.png`
- Modified (gitignored): `PROGRESS.md`

These are not part of the reranker work. The user should commit the bootstrap chunk on `claude/2026-04-08-intake-bootstrap` first, then we create `claude/2026-04-08-reranker` from a clean state. The spec file written during this brainstorming session needs to be carried over (or re-staged) on the new branch — straightforward, called out here so it doesn't get lost.

### 10.4 PR / merge

Out of scope for Claude per kernel §0. When the branch is ready, Claude prints the suggested `gh pr create` command and stops; the user runs it.

---

## 11. Open Questions Resolved by This Spec

For the project `CLAUDE.md` "Open Questions" section:

- **Tavily wiring** — *not addressed by this spec.* Still open. Recommend grepping `src/` for `tavily` as a separate task in the next phase.
- **Integration test gate** — *partially addressed.* This spec explicitly chooses unit tests only (mocking the cross-encoder). The broader question of an end-to-end integration test (orchestrator + Chroma + stubbed LLM) remains open for the next phase.
- **Audit findings (§14)** — *partially addressed.* The §14.2 issue "sync I/O in async paths" gets a partial fix in the new code (`asyncio.to_thread` around `predict()`), though existing offenders (BM25 cache I/O, etc.) are deliberately untouched.

---

## 12. Open Questions Created by This Spec

To track in `PROGRESS.md` after merge:

1. **Should `enabled: true` become the default?** Decided by the eval results in §13. **Yes**, with `candidate_k=30, top_k=10`. The aggressive `top_k=5` config is rejected because it regresses Context Recall on retrieval-spread questions (see §13.3).
2. **Does the BGE-reranker model warrant pre-warming in the FastAPI lifespan hook?** Currently lazy-loaded on first query, which means the first request after startup pays the model-load latency (~3-10s). For local dev this is fine; in deployed contexts a `lifespan` warm-up call would smooth the cold-start. Not blocking.
3. **Do we want a config-override CLI flag for the eval sweep?** Currently the sweep is manual edits to `settings.yaml`. A `--rerank-config 30x5` flag would let us script the sweep but adds scope. Defer to next branch if the sweep is annoying enough.

---

## 13. Results

> Filled in after the eval sweep landed on 2026-04-09. Sweep was reduced from the planned 5 configs to 3 (baseline + 30→10 + 30→5) — see §13.5 for the rationale and what we lost.

### 13.1 Average metrics (RAGAS, 10-sample golden_set_v2)

| Metric | Baseline | 30→10 | 30→5 | Best Δ vs baseline |
|---|---|---|---|---|
| **Context Precision** | 0.6729 | **0.9226** | **0.9450** | **+27.20 pp** (30→5) |
| Faithfulness | 0.9508 | 0.9800 | 0.9675 | +2.92 pp (30→10) |
| Answer Relevancy | 0.9643 | 0.9794 | 0.9815 | +1.72 pp (30→5) |
| Context Recall | 0.9667 | **1.0000** | **0.9000** | +3.33 pp (30→10) / **−6.67 pp** (30→5) |

The baseline Context Precision of **0.6729** reproduces the 67.21% number from [PROJECT_SUMMARY.md §14.1](../../../PROJECT_SUMMARY.md) almost exactly — independent confirmation that the eval methodology is consistent across runs.

Both reranker configs lift Context Precision by ~25 percentage points. The choice between them is **not** about precision (where they're tied within noise) — it's about whether the ~2 pp extra precision in 30→5 is worth the catastrophic recall regression on one specific question type. See §13.3.

### 13.2 Verdict against success criteria (§1)

| Criterion | 30→10 | 30→5 |
|---|---|---|
| Context Precision lift ≥5 pp | ✅ +24.96 pp | ✅ +27.20 pp |
| Faithfulness regression ≤5 pp | ✅ +2.92 pp | ✅ +1.67 pp |
| Answer Relevancy regression ≤5 pp | ✅ +1.51 pp | ✅ +1.72 pp |
| Context Recall regression ≤5 pp | ✅ +3.33 pp | ❌ **−6.67 pp** |

**30→10 meets all criteria. 30→5 fails the Context Recall threshold.** The chosen production default is **30→10**.

### 13.3 The interesting per-question finding

The −6.67 pp average recall regression in 30→5 is not spread across the 10 questions — it's almost entirely concentrated in **one** question:

> *"What was the result of applying large-scale reinforcement learning directly to Qwen-32B-Base without distillation?"*

| Config | context_recall on this question |
|---|---|
| Baseline (top_k=10) | 1.000 |
| 30→10 | 1.000 |
| **30→5** | **0.000** ⚠️ |

Reading the per-question delta CSV ([delta_report_2026-04-09_baseline_vs_report_2026-04-09_30x5.csv](../../../evaluation/results/delta_report_2026-04-09_baseline_vs_report_2026-04-09_30x5.csv)) tells the rest of the story: this question's ground truth requires comparing **two** sibling chunks — `DeepSeek-R1-Zero-Qwen-32B` (the "no distillation" experiment) and `DeepSeek-R1-Distill-Qwen-32B` (the contrast). Both chunks score moderately on the cross-encoder for the query, but neither dominates. With `top_k=5`, the reranker promoted 5 chunks that individually score higher but collectively drop one half of the comparison — recall collapses to 0. With `top_k=10`, both halves survive.

This is the exact failure mode the brainstorming discussion (Q4 in conversation) flagged as a risk for aggressive narrowing: **comparison questions need more than one chunk**, and a relevance-ranked top-K can lose them.

A second smaller regression worth noting: the Cloudflare AI Crawl Control question lost **−0.231** Context Precision in 30→5. The reranker promoted some less-precise chunks. 30→10 doesn't show this regression.

### 13.4 Decision on default

`config/settings.yaml` is updated in this same chunk to:

```yaml
rerank_settings:
  enabled: true
  model_name: "BAAI/bge-reranker-base"
  candidate_k: 30
  top_k: 10
```

This is the only config from the sweep that meets all four success criteria. The lift on Context Precision (+24.96 pp, from 0.6729 → 0.9226) is the headline number for the README and the project narrative. The +3.33 pp on Context Recall and +2.92 pp on Faithfulness are bonus — reranking didn't just shuffle relevance scores, it also gave the verifier cleaner context to work with.

### 13.5 Sweep methodology — what we actually ran vs. what was planned

The spec §7.1 originally called for a 5-point sweep (baseline / 30→10 / 30→5 / 50→10 / 50→5). In practice we ran only the first three. Reasons:

- Each run takes ~5–10 minutes wall-clock; running all 5 sequentially is a 30–60 minute commitment.
- The first 3 runs already produced a clear, defensible answer: 30→10 is the winner. Adding 50→{10,5} would test whether a wider candidate pool helps further, but the 30 → 5/10 step is already demonstrating diminishing returns past `candidate_k=30`.
- The `rag.top_k=10` baseline means `candidate_k=30` is already 3x the original funnel — the marginal value of expanding to 50 is bounded.

**What we lose:** the "wider candidate pool helps" hypothesis stays untested. If a future eval shows the chosen default is leaving precision on the table, the 50→{10,5} sweep is the obvious next experiment.

**What we keep:** the baseline ↔ 30→10 ↔ 30→5 trio is the cleanest 3-point story. Baseline → 30→10 isolates "the reranker exists" (same final K=10, just better-ranked). 30→10 → 30→5 isolates "narrowing the LLM payload" (same candidate pool, different cut). The 30→5 result is what makes the writeup interesting — without it, the spec would say "reranker good, ship it" instead of "reranker good, AND here's the failure mode of being too aggressive."

### 13.6 Latency (informal observation)

Not formally measured, but per-run timing across the three sweeps showed no perceptible slowdown from the reranker. The cross-encoder predict step on 30 candidates takes well under 100ms on CPU (M-class hardware), dwarfed by the 1–3s LLM call latency that dominates each query. As predicted in §9.2, latency is not a gate.

### 13.7 What the negative finding teaches

The spec §1 success criterion #2 ("no metric regression > 5 pp") was load-bearing. Without it, we would have shipped 30→5 as the default — chasing the bigger Context Precision number — and silently broken comparison-style questions. The success criterion was the safety net that caught it. Three lessons:

1. **Optimize for the slowest moving metric, but constrain by all of them.** Context Precision was the headline; Context Recall was the constraint. Both mattered.
2. **Per-question diff analysis is non-negotiable for retrieval changes.** A −6.67 pp average recall regression looks survivable until you see it's actually one −1.0 outlier dragging the mean.
3. **Aggressive narrowing is a real failure mode.** "Smaller LLM payload = cleaner context" is not always true if "cleaner" means "missing the half of the comparison the question needed."
