"""Per-question RAGAS delta report.

Joins two slim report_*.csv files (baseline + reranked) on user_input
and emits a per-question delta table sorted by Context Precision lift
descending. The output CSV is committed alongside the baseline runs.

Usage:
    python evaluation/compare_runs.py \\
        evaluation/results/report_<baseline>.csv \\
        evaluation/results/report_<reranked>.csv

Writes:
    evaluation/results/delta_<baseline_stem>_vs_<reranked_stem>.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_METRIC_COLS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
]


def compare(baseline_path: Path, reranked_path: Path) -> pd.DataFrame:
    base = pd.read_csv(baseline_path)
    rerank = pd.read_csv(reranked_path)

    if "user_input" not in base.columns or "user_input" not in rerank.columns:
        raise SystemExit(
            "Both CSVs must have a 'user_input' column. "
            f"Baseline columns: {list(base.columns)}; "
            f"reranked columns: {list(rerank.columns)}."
        )

    merged = base.merge(
        rerank,
        on="user_input",
        suffixes=("_baseline", "_reranked"),
    )

    for metric in _METRIC_COLS:
        b_col = f"{metric}_baseline"
        r_col = f"{metric}_reranked"
        if b_col in merged.columns and r_col in merged.columns:
            merged[f"{metric}_delta"] = merged[r_col] - merged[b_col]

    sort_col = (
        "context_precision_delta"
        if "context_precision_delta" in merged.columns
        else None
    )
    if sort_col:
        merged = merged.sort_values(sort_col, ascending=False)

    return merged


def _summary(df: pd.DataFrame) -> str:
    lines = ["Average deltas (reranked - baseline):"]
    for metric in _METRIC_COLS:
        delta_col = f"{metric}_delta"
        if delta_col in df.columns:
            avg = df[delta_col].mean()
            sign = "+" if avg >= 0 else ""
            lines.append(f"  {metric:20s} {sign}{avg:.4f}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("reranked", type=Path)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("evaluation/results"),
        help="Where to write the delta CSV (default: %(default)s)",
    )
    args = parser.parse_args()

    if not args.baseline.exists():
        print(f"baseline not found: {args.baseline}", file=sys.stderr)
        return 1
    if not args.reranked.exists():
        print(f"reranked not found: {args.reranked}", file=sys.stderr)
        return 1

    df = compare(args.baseline, args.reranked)

    out_path = args.out_dir / f"delta_{args.baseline.stem}_vs_{args.reranked.stem}.csv"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(_summary(df))
    print(f"\nDelta CSV written to: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
