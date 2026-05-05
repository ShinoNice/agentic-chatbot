from src.workflow.memory import AgentState


def test_agent_state_has_claims_and_final_answer_keys():
    keys = set(AgentState.__annotations__.keys())
    assert "claims" in keys
    assert "final_answer" in keys
    assert "draft_answer" in keys  # legacy still present during rollout
