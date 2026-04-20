from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd
from openai import AsyncOpenAI
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.llms import llm_factory
from ragas.metrics import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)
from ragas.run_config import RunConfig

_ROOT_DIR = str(Path(__file__).resolve().parent.parent)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from src.api.dependencies import SystemManager  # noqa: E402
from src.core.config_loader import settings  # noqa: E402

logger = logging.getLogger(__name__)

_DEFAULT_DATASET = "evaluation/datasets/golden_set_v2.json"
_DEFAULT_MAX_SAMPLES = 10
_RESULTS_DIR = Path("evaluation/results")


async def _collect_samples(
    system: SystemManager,
    raw_data: list[dict],
    max_samples: int,
) -> list[SingleTurnSample]:
    samples: list[SingleTurnSample] = []

    for idx, item in enumerate(raw_data[:max_samples], start=1):
        question = item["question"]
        logger.info("Processing sample %d/%d: %s", idx, max_samples, question[:60])

        result = await system.query(question)

        context = [
            doc.page_content if hasattr(doc, "page_content") else str(doc)
            for doc in result.get("documents", [])
        ]

        samples.append(
            SingleTurnSample(
                user_input=question,
                response=result.get("draft_answer", "N/A"),
                retrieved_contexts=context,
                reference=item["ground_truth"],
            )
        )

    return samples


def _build_evaluator_llm():
    return llm_factory(
        "gpt-4o-mini",
        client=AsyncOpenAI(api_key=settings.openai_api_key),
    )


def _save_results(result, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M")

    df = result.to_pandas()

    # Keep question for identification, drop bulky text columns
    score_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    keep = ["user_input"] + [c for c in score_cols if c in df.columns]
    df_slim = df[keep]

    output_path = output_dir / f"report_{timestamp}.csv"
    df_slim.to_csv(output_path, index=False)

    # Also save the full dataset for debugging if needed
    full_path = output_dir / f"report_{timestamp}_full.csv"
    df.to_csv(full_path, index=False)

    return output_path


async def run_evaluation(
    dataset_path: str = _DEFAULT_DATASET,
    max_samples: int = _DEFAULT_MAX_SAMPLES,
) -> None:
    os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key)

    system = SystemManager()
    await system.try_connect_existing()
    if not system.is_ready:
        logger.error("Knowledge base not ready. Ingest documents first.")
        return

    with open(dataset_path, encoding="utf-8") as f:
        raw_data: list[dict] = json.load(f)
    logger.info("Loaded %d samples from %s", len(raw_data), dataset_path)

    samples = await _collect_samples(system, raw_data, max_samples)

    evaluator_llm = _build_evaluator_llm()
    metrics = [
        Faithfulness(llm=evaluator_llm),
        AnswerRelevancy(llm=evaluator_llm),
        ContextPrecision(llm=evaluator_llm),
        ContextRecall(llm=evaluator_llm),
    ]
    run_config = RunConfig(timeout=300, max_retries=10, max_workers=1)

    logger.info("Starting RAGAS evaluation with %d metric(s)", len(metrics))

    result = evaluate(
        dataset=EvaluationDataset(samples=samples),
        metrics=metrics,
        run_config=run_config,
        raise_exceptions=False,
    )

    report_path = _save_results(result, _RESULTS_DIR)
    logger.info("Report saved to %s", report_path)
    print(f"\nEvaluation complete. Summary:\n{result}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RAGAS evaluation against the Agentic RAG pipeline.",
    )
    parser.add_argument(
        "--dataset",
        default=_DEFAULT_DATASET,
        help="Path to the golden-set JSON file (default: %(default)s).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=_DEFAULT_MAX_SAMPLES,
        help="Max samples to evaluate (default: %(default)s).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    args = _parse_args()
    asyncio.run(run_evaluation(dataset_path=args.dataset, max_samples=args.max_samples))
