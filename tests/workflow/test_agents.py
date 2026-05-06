"""Unit tests for the LangGraph agents.

The LLM is mocked end-to-end: no OpenAI calls, no network. Each test
pins a specific failure mode or branch so regressions surface loudly.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from src.schemas.agent_schemas import (
    RelevanceResponse,
    RelevanceStatus,
)
from src.workflow.agents.relevance_checker import RelevanceCheckerAgent

# ── Shared helpers ────────────────────────────────────────────────────


def _chain_returning(payload):
    """Runnable that ignores input and returns ``payload``.

    Agents do ``prompt | self.structured_llm`` so the RHS must be a real
    Runnable — Pydantic rejects plain MagicMocks at chain-build time.
    """
    return RunnableLambda(lambda _inputs: payload)


def _chain_raising(exc: Exception):
    """Runnable that raises ``exc`` on invoke to simulate LLM/chain failure."""

    def _raise(_inputs):
        raise exc

    return RunnableLambda(_raise)


def _fake_llm_with_structured_output(payload):
    """Mock BaseLLM whose model.with_structured_output() returns a Runnable."""
    fake_model = MagicMock()
    fake_model.with_structured_output.return_value = _chain_returning(payload)
    llm = MagicMock()
    llm.get_model.return_value = fake_model
    return llm, fake_model


# ── RelevanceCheckerAgent ────────────────────────────────────────────


@pytest.fixture
def docs():
    return [
        Document(page_content="alpha content", metadata={"source": "a.pdf"}),
        Document(page_content="beta content", metadata={"source": "b.pdf"}),
    ]


@pytest.mark.parametrize(
    "status",
    [RelevanceStatus.CAN_ANSWER, RelevanceStatus.PARTIAL, RelevanceStatus.NO_MATCH],
)
async def test_relevance_checker_returns_each_status(docs, status):
    """Agent propagates whatever status the structured-output chain returns."""
    llm, _ = _fake_llm_with_structured_output(RelevanceResponse(status=status, reasoning="test"))
    agent = RelevanceCheckerAgent(llm)
    # Patch the chain to bypass the ChatPromptTemplate pipe-operator gymnastics.
    agent.structured_llm = _chain_returning(RelevanceResponse(status=status, reasoning="test"))

    result = await agent.check("question?", docs)
    assert result == status


async def test_relevance_checker_returns_no_match_for_empty_docs():
    """Zero documents short-circuits to NO_MATCH without an LLM call."""
    llm, _ = _fake_llm_with_structured_output(
        RelevanceResponse(status=RelevanceStatus.CAN_ANSWER, reasoning="wrong")
    )
    agent = RelevanceCheckerAgent(llm)

    result = await agent.check("question?", [])
    assert result == RelevanceStatus.NO_MATCH


async def test_relevance_checker_returns_no_match_on_llm_error(docs):
    """If the structured-output chain raises, degrade to NO_MATCH (safe default)."""
    llm, _ = _fake_llm_with_structured_output(None)
    agent = RelevanceCheckerAgent(llm)

    agent.structured_llm = _chain_raising(RuntimeError("LLM went boom"))

    result = await agent.check("question?", docs)
    assert result == RelevanceStatus.NO_MATCH


