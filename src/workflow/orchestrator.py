import asyncio
from typing import Any

from langgraph.graph import END, START, StateGraph

from src.core.config_loader import settings as app_settings
from src.core.exceptions import RerankerError
from src.core.logger import logger
from src.engines.openai_client import OpenAIClient
from src.mcp.client import audit_log, guardrails_redact, guardrails_scan
from src.retrieval.hybrid_search import HybridSearcher, ensure_chunk_ids
from src.retrieval.reranker import BGEReranker
from src.schemas.agent_schemas import GuardrailsReport, RelevanceStatus
from src.schemas.claim_schemas import ClaimStatus
from src.workflow.agents.claim_drafter import ClaimDrafter
from src.workflow.agents.claim_verifier import ClaimVerifier
from src.workflow.agents.relevance_checker import (
    RelevanceCheckerAgent as RelevanceChecker,
)
from src.workflow.agents.researcher import ResearchAgent
from src.workflow.agents.verifier import VerificationAgent
from src.workflow.claim_pipeline.repair import repair_claims as _repair_claims
from src.workflow.claim_pipeline.render import render_final_answer
from src.workflow.memory import AgentState


class RAGOrchestrator:
    """Orchestrates the multi-agent RAG workflow with a self-correcting feedback loop."""

    def __init__(self, searcher: HybridSearcher):
        self.searcher = searcher
        self.llm_engine = OpenAIClient()

        self.relevance_checker = RelevanceChecker(self.llm_engine)
        self.researcher = ResearchAgent(self.llm_engine)
        self.verifier = VerificationAgent(self.llm_engine)

        self.claim_drafter = ClaimDrafter(
            self.llm_engine,
            max_claims=app_settings.claim_pipeline.drafter_max_claims,
        )
        self.claim_verifier = ClaimVerifier(
            self.llm_engine,
            concurrency=app_settings.claim_pipeline.verify_concurrency,
        )

        self.reranker = (
            BGEReranker(app_settings.rerank.model_name) if app_settings.rerank.enabled else None
        )

        self.app = self._build_graph()

    def _build_graph(self):
        """Define nodes and self-correcting edges of the workflow."""
        workflow = StateGraph(AgentState)

        workflow.add_node("guardrails_input", self.node_guardrails_input)
        workflow.add_node("retrieve", self.node_retrieve)
        workflow.add_node("rerank", self.node_rerank)
        workflow.add_node("check_relevance", self.node_check_relevance)
        workflow.add_node("guardrails_output", self.node_guardrails_output)

        workflow.add_edge(START, "guardrails_input")
        workflow.add_edge("guardrails_input", "retrieve")
        workflow.add_edge("retrieve", "rerank")
        workflow.add_edge("rerank", "check_relevance")

        if app_settings.claim_pipeline.enabled:
            workflow.add_node("draft_claims", self.node_draft_claims)
            workflow.add_node("verify_claims", self.node_verify_claims)
            workflow.add_node("repair_claims", self.node_repair_claims)
            workflow.add_node("finalize", self.node_finalize)

            workflow.add_conditional_edges(
                "check_relevance",
                self.decide_after_relevance,
                {"proceed": "draft_claims", "stop": "finalize"},
            )
            workflow.add_edge("draft_claims", "guardrails_output")
            workflow.add_edge("guardrails_output", "verify_claims")
            workflow.add_conditional_edges(
                "verify_claims",
                self.decide_after_claim_verification,
                {"finalize": "finalize", "repair": "repair_claims"},
            )
            workflow.add_edge("repair_claims", "verify_claims")
            workflow.add_edge("finalize", END)
        else:
            workflow.add_node("research", self.node_research)
            workflow.add_node("verify", self.node_verify)

            workflow.add_conditional_edges(
                "check_relevance",
                self.decide_after_relevance,
                {"proceed": "research", "stop": END},
            )
            workflow.add_edge("research", "guardrails_output")
            workflow.add_edge("guardrails_output", "verify")
            workflow.add_conditional_edges(
                "verify",
                self.decide_after_verification,
                {"finalize": END, "retry": "research"},
            )

        return workflow.compile()

    # -- Guardrails Nodes --

    async def node_guardrails_input(self, state: AgentState) -> dict[str, Any]:
        logger.info("--- NODE: GUARDRAILS INPUT ---")
        session_id = state.get("audit_session_id", "")

        if not app_settings.mcp.guardrails.enabled or not app_settings.mcp.guardrails.scan_input:
            await audit_log(
                session_id,
                "query_received",
                "guardrails_input",
                {"question_length": len(state["question"]), "has_pii": False},
            )
            return {}

        scan_result = await guardrails_scan(state["question"])
        report = GuardrailsReport(
            input_pii_found=scan_result.has_pii,
            input_redactions=len(scan_result.detections),
            detections=list(scan_result.detections),
        )

        updates: dict[str, Any] = {"guardrails_report": report}

        if scan_result.has_pii:
            redacted = await guardrails_redact(state["question"])
            updates["question"] = redacted.redacted_text
            logger.warning(
                f"PII detected in input query: {len(scan_result.detections)} detection(s). "
                "Query redacted before retrieval."
            )
            await audit_log(
                session_id,
                "pii_detected",
                "guardrails_input",
                {
                    "scan_target": "input",
                    "pii_types": [d.pii_type for d in scan_result.detections],
                    "count": len(scan_result.detections),
                },
            )
        else:
            await audit_log(
                session_id,
                "query_received",
                "guardrails_input",
                {"question_length": len(state["question"]), "has_pii": False},
            )

        return updates

    async def node_guardrails_output(self, state: AgentState) -> dict[str, Any]:
        logger.info("--- NODE: GUARDRAILS OUTPUT ---")
        session_id = state.get("audit_session_id", "")
        draft = state.get("draft_answer", "")

        if (
            not app_settings.mcp.guardrails.enabled
            or not app_settings.mcp.guardrails.scan_output
            or not draft
        ):
            return {}

        scan_result = await guardrails_scan(draft)

        updates: dict[str, Any] = {}

        # Build a new report preserving input detections from earlier
        existing = state.get("guardrails_report")
        if existing is not None:
            new_report = GuardrailsReport(
                input_pii_found=existing.input_pii_found,
                input_redactions=existing.input_redactions,
                output_pii_found=scan_result.has_pii,
                output_redactions=len(scan_result.detections),
                detections=list(existing.detections) + list(scan_result.detections),
            )
        else:
            new_report = GuardrailsReport(
                output_pii_found=scan_result.has_pii,
                output_redactions=len(scan_result.detections),
                detections=list(scan_result.detections),
            )

        if scan_result.has_pii:
            redacted = await guardrails_redact(draft)
            updates["draft_answer"] = redacted.redacted_text
            logger.warning(
                f"PII detected in draft answer: {len(scan_result.detections)} detection(s). "
                "Answer redacted before verification."
            )
            await audit_log(
                session_id,
                "pii_redacted",
                "guardrails_output",
                {
                    "redactions": len(scan_result.detections),
                    "strategy": app_settings.mcp.guardrails.redaction_strategy,
                },
            )

        updates["guardrails_report"] = new_report
        return updates

    # -- Original Nodes (with audit logging) --

    async def node_retrieve(self, state: AgentState) -> dict[str, Any]:
        logger.info("--- NODE: RETRIEVAL ---")
        retriever = await self.searcher.aget_retriever()
        docs = await retriever.ainvoke(state["question"])
        docs = ensure_chunk_ids(docs)

        session_id = state.get("audit_session_id", "")
        await audit_log(
            session_id,
            "documents_retrieved",
            "retrieve",
            {
                "candidate_count": len(docs),
                "doc_ids": [d.metadata.get("chunk_hash", "")[:12] for d in docs[:10]],
            },
        )

        return {"candidate_documents": docs}

    async def node_rerank(self, state: AgentState) -> dict[str, Any]:
        logger.info("--- NODE: RERANK ---")
        candidates = state["candidate_documents"]
        session_id = state.get("audit_session_id", "")

        if not app_settings.rerank.enabled or self.reranker is None:
            await audit_log(
                session_id,
                "documents_reranked",
                "rerank",
                {
                    "input_count": len(candidates),
                    "output_count": len(candidates),
                    "reranker": "disabled",
                },
            )
            return {"documents": candidates}

        try:
            reranked = await asyncio.to_thread(
                self.reranker.rerank,
                state["question"],
                candidates,
                app_settings.rerank.top_k,
            )
            logger.info(f"Reranked {len(candidates)} → {len(reranked)} chunks")

            await audit_log(
                session_id,
                "documents_reranked",
                "rerank",
                {"input_count": len(candidates), "output_count": len(reranked)},
            )

            return {"documents": reranked}
        except RerankerError as e:
            fallback = candidates[: app_settings.rerank.top_k]
            logger.error(
                f"Reranker failed, falling back to first {len(fallback)} "
                f"candidates (untruncated pool was {len(candidates)}): {e}"
            )
            return {"documents": fallback}

    async def node_check_relevance(self, state: AgentState) -> dict[str, Any]:
        logger.info("--- NODE: RELEVANCE AUDIT ---")
        status = await self.relevance_checker.check(state["question"], state["documents"])

        session_id = state.get("audit_session_id", "")
        await audit_log(
            session_id,
            "relevance_checked",
            "check_relevance",
            {"status": status.value if isinstance(status, RelevanceStatus) else str(status)},
        )

        return {"relevance_status": status}

    async def node_research(self, state: AgentState) -> dict[str, Any]:
        logger.info("--- NODE: RESEARCH & DRAFTING ---")
        current_iter = state.get("iterations", 0)
        answer = await self.researcher.generate(state["question"], state["documents"])

        session_id = state.get("audit_session_id", "")
        await audit_log(
            session_id,
            "answer_generated",
            "research",
            {"iteration": current_iter + 1, "answer_length": len(answer)},
        )

        return {"draft_answer": answer, "iterations": current_iter + 1}

    async def node_verify(self, state: AgentState) -> dict[str, Any]:
        logger.info("--- NODE: VERIFICATION AUDIT ---")
        report = await self.verifier.verify(
            state["question"],
            state["draft_answer"],
            state["documents"],
        )

        session_id = state.get("audit_session_id", "")
        await audit_log(
            session_id,
            "verification_completed",
            "verify",
            {
                "supported": report.supported,
                "unsupported_count": len(report.unsupported_claims),
                "contradiction_count": len(report.contradictions),
            },
        )

        return {"verification": report}

    # -- Claim-grounded pipeline nodes --

    async def node_draft_claims(self, state: AgentState) -> dict[str, Any]:
        logger.info("--- NODE: DRAFT CLAIMS ---")
        claim_set = await self.claim_drafter.draft(state["question"], state["documents"])
        # Mirror text into draft_answer so the existing guardrails_output node has
        # something to scan.
        joined = " ".join(c.text for c in claim_set.claims)
        await audit_log(
            state.get("audit_session_id", ""),
            "claims_drafted",
            "draft_claims",
            {"count": len(claim_set.claims), "claim_ids": [c.id for c in claim_set.claims]},
        )
        return {"claims": list(claim_set.claims), "draft_answer": joined}

    async def node_verify_claims(self, state: AgentState) -> dict[str, Any]:
        logger.info("--- NODE: VERIFY CLAIMS ---")
        pending = [
            c for c in (state.get("claims") or []) if c.status == ClaimStatus.PENDING
        ]
        verified = await self.claim_verifier.verify(
            state["question"], pending, state["documents"]
        )
        by_id = {c.id: c for c in (state.get("claims") or [])}
        for c in verified:
            by_id[c.id] = c
        await audit_log(
            state.get("audit_session_id", ""),
            "claims_verified",
            "verify_claims",
            {"results": [{"id": c.id, "status": c.status.value} for c in verified]},
        )
        return {"claims": list(by_id.values())}

    async def node_repair_claims(self, state: AgentState) -> dict[str, Any]:
        logger.info("--- NODE: REPAIR CLAIMS ---")
        updated, augmented = await _repair_claims(
            question=state["question"],
            claims=state.get("claims") or [],
            chunks=state["documents"],
            drafter=self.claim_drafter,
            searcher=self.searcher,
            max_attempts=app_settings.claim_pipeline.max_repair_rounds,
        )
        repaired = [c for c in updated if c.attempts > 0]
        await audit_log(
            state.get("audit_session_id", ""),
            "claims_repaired",
            "repair_claims",
            {"repaired_ids": [c.id for c in repaired]},
        )
        return {"claims": updated, "documents": augmented}

    async def node_finalize(self, state: AgentState) -> dict[str, Any]:
        logger.info("--- NODE: FINALIZE ---")
        claims = state.get("claims") or []
        return {"final_answer": render_final_answer(claims)}

    # -- Routing --

    def decide_after_relevance(self, state: AgentState) -> str:
        if state["relevance_status"] == RelevanceStatus.NO_MATCH:
            logger.warning("Retrieved context is irrelevant. Terminating.")
            return "stop"
        return "proceed"

    def decide_after_verification(self, state: AgentState) -> str:
        report = state["verification"]

        if report.supported:
            return "finalize"

        if state["iterations"] >= app_settings.app.max_iterations:
            logger.error("Max retries reached. Returning best-effort answer.")
            return "finalize"

        logger.info(f"Hallucination found (iter {state['iterations']}). Retrying...")
        return "retry"

    def decide_after_claim_verification(self, state: AgentState) -> str:
        claims = state.get("claims") or []
        bad = [
            c
            for c in claims
            if c.status in (ClaimStatus.UNSUPPORTED, ClaimStatus.CONTRADICTED)
        ]
        repairable = [
            c for c in bad if c.attempts < app_settings.claim_pipeline.max_repair_rounds
        ]
        if repairable:
            logger.info(f"decide: routing {len(repairable)} claim(s) to repair")
            return "repair"
        return "finalize"

    async def run(self, question: str, session_id: str = ""):
        """Execute the state machine for a given question."""
        initial_state = {
            "question": question,
            "candidate_documents": [],
            "documents": [],
            "iterations": 0,
            "guardrails_report": None,
            "audit_session_id": session_id,
            "claims": None,
            "final_answer": None,
        }
        budget = (
            app_settings.claim_pipeline.total_budget_seconds
            if app_settings.claim_pipeline.enabled
            else None
        )
        try:
            if budget:
                result = await asyncio.wait_for(
                    self.app.ainvoke(initial_state), timeout=budget
                )
            else:
                result = await self.app.ainvoke(initial_state)
        except asyncio.TimeoutError:
            logger.warning(
                f"Run exceeded budget {budget}s; returning truncated state."
            )
            result = {
                **initial_state,
                "truncated": True,
                "final_answer": render_final_answer([]),
            }

        await audit_log(
            session_id,
            "answer_delivered",
            "orchestrator",
            {
                "final_status": "answered",
                "total_iterations": result.get("iterations", 0),
            },
        )

        return result

    async def astream_run(self, question: str, session_id: str = ""):
        """Yield (event_type, payload) tuples as the graph executes.

        Emits ('step', {...}) per node completion so the UI can show progress,
        then ('result', final_state) once the graph finishes.
        """
        initial_state = {
            "question": question,
            "candidate_documents": [],
            "documents": [],
            "iterations": 0,
            "guardrails_report": None,
            "audit_session_id": session_id,
            "claims": None,
            "final_answer": None,
        }

        accumulated: dict[str, Any] = {}

        deadline = (
            asyncio.get_event_loop().time() + app_settings.claim_pipeline.total_budget_seconds
            if app_settings.claim_pipeline.enabled
            else None
        )

        async for chunk in self.app.astream(initial_state, stream_mode="updates"):
            if deadline and asyncio.get_event_loop().time() > deadline:
                logger.warning("astream exceeded budget; emitting truncated result.")
                yield "result", {
                    **accumulated,
                    "truncated": True,
                    "answer": render_final_answer(accumulated.get("claims") or []),
                }
                return
            for node_name, delta in chunk.items():
                if not isinstance(delta, dict):
                    # Try to coerce pydantic models / objects with dict()/model_dump()
                    coerced: dict[str, Any] | None = None
                    if hasattr(delta, "model_dump"):
                        try:
                            coerced = delta.model_dump()
                        except Exception:
                            coerced = None
                    elif hasattr(delta, "dict"):
                        try:
                            coerced = delta.dict()
                        except Exception:
                            coerced = None
                    if not isinstance(coerced, dict):
                        logger.warning(
                            f"astream_run: dropping non-dict delta from node "
                            f"{node_name!r} (type={type(delta).__name__})"
                        )
                        continue
                    delta = coerced
                accumulated.update(delta)
                # Per-claim events when this delta carries claims.
                if "claims" in delta and isinstance(delta["claims"], list):
                    if node_name == "draft_claims":
                        for claim in delta["claims"]:
                            yield "claim_drafted", {"claim": claim.model_dump()}
                    elif node_name == "verify_claims":
                        for claim in delta["claims"]:
                            if claim.status != ClaimStatus.PENDING:
                                yield "claim_verified", {
                                    "claim_id": claim.id,
                                    "status": claim.status.value,
                                    "note": claim.verifier_note,
                                }
                    elif node_name == "repair_claims":
                        for claim in delta["claims"]:
                            if claim.attempts > 0:
                                yield "claim_repaired", {
                                    "claim_id": claim.id,
                                    "status": claim.status.value,
                                }
                extra: dict[str, Any] = {}
                if node_name == "retrieve":
                    extra["doc_count"] = len(delta.get("candidate_documents") or [])
                elif node_name == "verify" and "verification" in delta:
                    report = delta["verification"]
                    extra["supported"] = bool(getattr(report, "supported", False))
                elif node_name == "research":
                    extra["iteration"] = accumulated.get("iterations", 0)
                yield "step", {"node": node_name, "status": "completed", **extra}

        result_payload: dict[str, Any] = dict(accumulated)
        result_payload["answer"] = (
            accumulated.get("final_answer") or accumulated.get("draft_answer") or ""
        )
        if accumulated.get("claims"):
            result_payload["claims"] = [
                c.model_dump() if hasattr(c, "model_dump") else c
                for c in accumulated["claims"]
            ]
        yield "result", result_payload

        await audit_log(
            session_id,
            "answer_delivered",
            "orchestrator",
            {
                "final_status": "answered",
                "total_iterations": accumulated.get("iterations", 0),
            },
        )
