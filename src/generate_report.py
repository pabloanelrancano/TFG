# src/generate_report.py
# Pablo Anel Rancano - TFG HAR
"""Combine per-pipeline summary CSVs into a global comparison report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))


def find_summary_csvs(results_dir: Path) -> List[Path]:
    """Find all summary_*.csv in results/."""
    csvs = sorted(results_dir.rglob("summary_*.csv"))
    return csvs


def load_and_tag(csv_path: Path) -> pd.DataFrame:
    """Load a summary CSV and tag it with the pipeline name."""
    df = pd.read_csv(csv_path)
    pipeline = csv_path.parent.name
    if pipeline == csv_path.parent.name:
        pipeline = csv_path.stem.replace("summary_", "")
    df.insert(0, "Pipeline", pipeline)
    return df


def generate_comparison_report(results_dir: Path) -> None:
    """Build global comparison (CSV + Markdown) from pipeline summaries."""
    csvs = find_summary_csvs(results_dir)

    if not csvs:
        print(f"No summary CSVs found in {results_dir.resolve()}")
        print("Run experiments first: python src/run_experiments.py")
        return

    print(f"Found {len(csvs)} summary file(s):")
    for c in csvs:
        print(f"  - {c.relative_to(results_dir)}")

    dfs = [load_and_tag(c) for c in csvs]
    combined = pd.concat(dfs, ignore_index=True)

    if "comparison_all_pipelines.csv" in [c.name for c in csvs]:
        combined = combined.drop_duplicates(subset=["Pipeline", "Model"], keep="first")

    core_cols = ["Pipeline", "Model", "Features", "Test Accuracy",
                 "Test F1 (macro)", "Test F1 (weighted)", "CV Mean", "CV Std", "Time (s)"]
    core_cols = [c for c in core_cols if c in combined.columns]
    combined = combined[core_cols].sort_values(
        ["Test Accuracy"], ascending=False
    ).reset_index(drop=True)

    csv_out = results_dir / "comparison_all_pipelines.csv"
    combined.to_csv(csv_out, index=False)
    print(f"\nCSV -> {csv_out}")

    md_out = results_dir / "comparison_all_pipelines.md"
    lines = [
        "# TFG HAR – Cross-Pipeline Comparison Report",
        "",
        "## All Results (sorted by Test Accuracy)",
        "",
    ]

    header = "| " + " | ".join(core_cols) + " |"
    sep = "| " + " | ".join("---" for _ in core_cols) + " |"
    lines.append(header)
    lines.append(sep)

    for _, row in combined.iterrows():
        vals = []
        for c in core_cols:
            v = row[c]
            if isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")

    lines.append("")

    if len(combined) > 0:
        best = combined.iloc[0]
        lines.append("## Highlights")
        lines.append("")
        lines.append(f"- **Best Test Accuracy:** {best['Model']} on "
                     f"*{best['Pipeline']}* → **{best['Test Accuracy']:.4f}**")

        if "CV Mean" in combined.columns:
            best_cv = combined.sort_values("CV Mean", ascending=False).iloc[0]
            lines.append(f"- **Best CV Mean:** {best_cv['Model']} on "
                         f"*{best_cv['Pipeline']}* → **{best_cv['CV Mean']:.4f}** "
                         f"(±{best_cv['CV Std']:.4f})")

        lines.append("")

        lines.append("## Per-Pipeline Best")
        lines.append("")
        for pipeline, grp in combined.groupby("Pipeline"):
            best_p = grp.sort_values("Test Accuracy", ascending=False).iloc[0]
            lines.append(f"- **{pipeline}**: {best_p['Model']} "
                         f"(acc={best_p['Test Accuracy']:.4f}, "
                         f"cv={best_p['CV Mean']:.4f}±{best_p['CV Std']:.4f})")
        lines.append("")

        lines.append("## CV Stability (lowest std = most stable)")
        lines.append("")
        if "CV Std" in combined.columns:
            stable = combined.sort_values("CV Std", ascending=True)
            for _, row in stable.head(5).iterrows():
                lines.append(f"- {row['Pipeline']}/{row['Model']}: "
                             f"CV={row['CV Mean']:.4f}±{row['CV Std']:.4f}")
        lines.append("")

    md_out.write_text("\n".join(lines), encoding="utf-8")
    print(f"MD  -> {md_out}")

    print("\n" + "=" * 70)
    print(combined.to_string(index=False))
    print("=" * 70)


def main() -> None:
    from config import get_config, add_common_args, apply_cli_overrides, results_dir as _results_dir

    parser = argparse.ArgumentParser(
        description="Generate cross-pipeline comparison report."
    )
    add_common_args(parser)
    args = parser.parse_args()
    cfg = apply_cli_overrides(get_config(), args)

    generate_comparison_report(_results_dir(cfg))


if __name__ == "__main__":
    main()
