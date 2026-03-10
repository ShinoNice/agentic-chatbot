from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import List

import pandas as pd
from langchain_openai import ChatOpenAI
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.llms import LangchainLLMWrapper
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

logger = logging.getLogger(__name__)

_DEFAULT_DATASET = "evaluation/datasets/golden_set_v1.json"
_DEFAULT_MAX_SAMPLES = 10
_RESULTS_DIR = Path("evaluation/results")
_TPM_DELAY_SECONDS = 2.0


class ThrottledChatOpenAI(ChatOpenAI):
    """Injects a per-call delay to stay within TPM rate limits."""

    def _generate(self, *args, **kwargs):
        time.sleep(_TPM_DELAY_SECONDS)
        return super()._generate(*args, **kwargs)


async def _collect_samples(
    system: SystemManager,
    raw_data: List[dict],
    max_samples: int,
) -> List[SingleTurnSample]:
    samples: List[SingleTurnSample] = []

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


def _build_evaluator_llm() -> LangchainLLMWrapper:
    base_llm = ThrottledChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        model_kwargs={"response_format": {"type": "json_object"}},
    )
    return LangchainLLMWrapper(base_llm)


def _save_results(result, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
    output_path = output_dir / f"report_{timestamp}.csv"
    result.to_pandas().to_csv(output_path, index=False)
    return output_path


async def run_evaluation(
    dataset_path: str = _DEFAULT_DATASET,
    max_samples: int = _DEFAULT_MAX_SAMPLES,
) -> None:
    system = SystemManager()
    await system.try_connect_existing()
    if not system.is_ready:
        logger.error("Knowledge base not ready. Ingest documents first.")
        return

    with open(dataset_path, "r", encoding="utf-8") as f:
        raw_data: List[dict] = json.load(f)
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
