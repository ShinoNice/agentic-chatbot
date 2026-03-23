import asyncio
from pathlib import Path


async def generate_diagram():
    """Generate and save the workflow diagram as PNG."""

    print("Building workflow graph structure...")

    try:
        from langgraph.graph import StateGraph, END, START
        from src.workflow.memory import AgentState

        workflow = StateGraph(AgentState)

        # Mirror the actual nodes from orchestrator.py
        dummy = lambda x: x
        workflow.add_node("retrieve", dummy)
        workflow.add_node("check_relevance", dummy)
        workflow.add_node("research", dummy)
        workflow.add_node("verify", dummy)

        # Mirror the actual edges from orchestrator.py
        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "check_relevance")
        workflow.add_conditional_edges(
            "check_relevance",
            lambda x: "proceed",
            {"proceed": "research", "stop": END},
        )
        workflow.add_edge("research", "verify")
        workflow.add_conditional_edges(
            "verify",
            lambda x: "finalize",
            {"finalize": END, "retry": "research"},
        )

        app = workflow.compile()

        print("Generating Mermaid diagram...")

        try:
            mermaid_png = app.get_graph().draw_mermaid_png()
            output_path = Path("workflow_diagram.png")
            output_path.write_bytes(mermaid_png)
            print(f"✓ PNG diagram saved to: {output_path.absolute()}")

        except Exception as e:
            print(f"PNG generation failed: {e}")

    except Exception as e:
        print(f"Error generating diagram: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(generate_diagram())
