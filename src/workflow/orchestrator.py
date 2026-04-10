import asyncio
from typing import Any, Dict

from langgraph.graph import END, START, StateGraph

from src.workflow.agents.relevance_checker import (
    RelevanceCheckerAgent as RelevanceChecker,
)
from src.workflow.agents.researcher import ResearchAgent
from src.workflow.agents.verifier import VerificationAgent
from src.workflow.memory import AgentState
from src.core.config_loader import settings
from src.core.exceptions import RerankerError
from src.core.logger import logger
from src.engines.openai_client import OpenAIClient
from src.retrieval.hybrid_search import HybridSearcher
from src.retrieval.reranker import BGEReranker
from src.schemas.agent_schemas import RelevanceStatus


class RAGOrchestrator:
    """Orchestrates the multi-agent RAG workflow with a self-correcting feedback loop."""

    def __init__(self, searcher: HybridSearcher):
        self.searcher = searcher
        self.llm_engine = OpenAIClient()

        self.relevance_checker = RelevanceChecker(self.llm_engine)
        self.researcher = ResearchAgent(self.llm_engine)
        self.verifier = VerificationAgent(self.llm_engine)

        self.reranker = (
            BGEReranker(settings.rerank.model_name) if settings.rerank.enabled else None
        )

        self.app = self._build_graph()

    def _build_graph(self):
        """Define nodes and self-correcting edges of the workflow."""
        workflow = StateGraph(AgentState)

        workflow.add_node("retrieve", self.node_retrieve)
        workflow.add_node("rerank", self.node_rerank)
        workflow.add_node("check_relevance", self.node_check_relevance)
        workflow.add_node("research", self.node_research)
        workflow.add_node("verify", self.node_verify)

        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "rerank")
        workflow.add_edge("rerank", "check_relevance")

        workflow.add_conditional_edges(
            "check_relevance",
            self.decide_after_relevance,
            {"proceed": "research", "stop": END},
        )

        workflow.add_edge("research", "verify")

        workflow.add_conditional_edges(
            "verify",
            self.decide_after_verification,
            {"finalize": END, "retry": "research"},
        )

        return workflow.compile()

    # -- Nodes --

    async def node_retrieve(self, state: AgentState) -> Dict[str, Any]:
        logger.info("--- NODE: RETRIEVAL ---")
        retriever = self.searcher.get_retriever()
        docs = await retriever.ainvoke(state["question"])
        return {"candidate_documents": docs}

    async def node_rerank(self, state: AgentState) -> Dict[str, Any]:
        logger.info("--- NODE: RERANK ---")
        candidates = state["candidate_documents"]

        if not settings.rerank.enabled or self.reranker is None:
            # Disabled path: forward all candidates. retrieve already returned
            # rag.top_k docs (not candidate_k), so the LLM payload is unchanged
            # from pre-reranker behavior.
            return {"documents": candidates}

        try:
            reranked = await asyncio.to_thread(
                self.reranker.rerank,
                state["question"],
                candidates,
                settings.rerank.top_k,
            )
            logger.info(f"Reranked {len(candidates)} → {len(reranked)} chunks")
            return {"documents": reranked}
        except RerankerError as e:
            # Graceful degradation: a broken reranker should not 500 the request.
            # Truncate to top_k anyway so a silent reranker failure does not
            # quietly triple the LLM token bill — better to ship a bounded
            # payload of arbitrary candidates than the full candidate_k pool.
            fallback = candidates[: settings.rerank.top_k]
            logger.error(
                f"Reranker failed, falling back to first {len(fallback)} "
                f"candidates (untruncated pool was {len(candidates)}): {e}"
            )
            return {"documents": fallback}

    async def node_check_relevance(self, state: AgentState) -> Dict[str, Any]:
        logger.info("--- NODE: RELEVANCE AUDIT ---")
        status = await self.relevance_checker.check(
            state["question"], state["documents"]
        )
        return {"relevance_status": status}

    async def node_research(self, state: AgentState) -> Dict[str, Any]:
        logger.info("--- NODE: RESEARCH & DRAFTING ---")
        current_iter = state.get("iterations", 0)
        answer = await self.researcher.generate(state["question"], state["documents"])
        return {"draft_answer": answer, "iterations": current_iter + 1}

    async def node_verify(self, state: AgentState) -> Dict[str, Any]:
        logger.info("--- NODE: VERIFICATION AUDIT ---")
        report = await self.verifier.verify(
            state["question"],
            state["draft_answer"],
            state["documents"],
        )
        return {"verification": report}

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

        if state["iterations"] >= settings.app.max_iterations:
            logger.error("Max retries reached. Returning best-effort answer.")
            return "finalize"

        logger.info(f"Hallucination found (iter {state['iterations']}). Retrying...")
        return "retry"

    async def run(self, question: str):
        """Execute the state machine for a given question."""
        initial_state = {
            "question": question,
            "candidate_documents": [],
            "documents": [],
            "iterations": 0,
        }
        return await self.app.ainvoke(initial_state)
