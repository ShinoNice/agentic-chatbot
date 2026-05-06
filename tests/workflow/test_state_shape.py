from src.workflow.memory import AgentState


def test_agent_state_has_claims_and_final_answer_keys():
    keys = set(AgentState.__annotations__.keys())
    assert "claims" in keys
    assert "final_answer" in keys
    # Legacy fields removed in T19 cutover.
    assert "draft_answer" not in keys
    assert "verification" not in keys
    assert "iterations" not in keys
