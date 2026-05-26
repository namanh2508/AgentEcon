#!/usr/bin/env python3
"""
Generate figures from experiment results.
"""

import json
import logging
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_figure3(df: pd.DataFrame, output_dir: Path):
    """Generate Figure 3: Honest Reporting Fraction."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # tau effect
    tau_means = df.groupby("tau")["honest_reporting_fraction"].agg(["mean", "std"])
    ax = axes[0]
    tau_means["mean"].plot(kind="bar", ax=ax, yerr=tau_means["std"], capsize=3)
    ax.set_xlabel("tau (rationality parameter)")
    ax.set_ylabel("Honest Reporting Fraction")
    ax.set_title("Effect of tau on Honest Reporting")
    ax.set_ylim(0, 1)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Random baseline")
    ax.legend()

    # lambda effect
    lambda_means = df.groupby("lambda")["honest_reporting_fraction"].agg(["mean", "std"])
    ax = axes[1]
    lambda_means["mean"].plot(kind="bar", ax=ax, yerr=lambda_means["std"], capsize=3, color="green")
    ax.set_xlabel("lambda (reputation weight)")
    ax.set_ylabel("Honest Reporting Fraction")
    ax.set_title("Effect of lambda on Honest Reporting")
    ax.set_ylim(0, 1)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.legend()

    plt.tight_layout()
    output_file = output_dir / "figure3_honest_reporting.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "figure3_honest_reporting.pdf", bbox_inches="tight")
    logger.info(f"Saved Figure 3 to {output_file}")


def generate_figure4(df: pd.DataFrame, output_dir: Path):
    """Generate Figure 4: Welfare Gap."""
    fig, ax = plt.subplots(figsize=(10, 6))

    lambda_welfare = df.groupby("lambda")["welfare_gap"].agg(["mean", "std"])
    lambda_welfare["mean"].plot(kind="bar", ax=ax, yerr=lambda_welfare["std"], capsize=3, color="orange")
    ax.set_xlabel("lambda (reputation weight)")
    ax.set_ylabel("Welfare Gap")
    ax.set_title("Effect of lambda on Welfare Gap")
    ax.set_ylim(0, None)

    plt.tight_layout()
    output_file = output_dir / "figure4_welfare_gap.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "figure4_welfare_gap.pdf", bbox_inches="tight")
    logger.info(f"Saved Figure 4 to {output_file}")


def generate_figure5(df: pd.DataFrame, output_dir: Path):
    """Generate Figure 5: Multi-LLM Consistency."""
    fig, ax = plt.subplots(figsize=(12, 6))

    model_means = df.groupby("model")["honest_reporting_fraction"].agg(["mean", "std", "count"])
    models = model_means.index.tolist()
    x = np.arange(len(models))
    means = model_means["mean"].values
    stds = model_means["std"].values

    ax.bar(x, means, yerr=stds, capsize=5, color=["blue", "green", "red"][:len(models)])
    ax.set_xticks(x)
    ax.set_xticklabels([m.split("/")[-1] for m in models], rotation=15, ha="right")
    ax.set_ylabel("Honest Reporting Fraction")
    ax.set_title("Multi-LLM Consistency Analysis")
    ax.set_ylim(0, 1)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    output_file = output_dir / "figure5_multi_llm.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "figure5_multi_llm.pdf", bbox_inches="tight")
    logger.info(f"Saved Figure 5 to {output_file}")


def generate_figure6(df: pd.DataFrame, output_dir: Path):
    """Generate Figure 6: Prompt Injection Robustness."""
    if "eta_adv" not in df.columns:
        logger.warning("Skipping Figure 6: results file has no eta_adv column")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    eta_means = df.groupby("eta_adv")["honest_reporting_fraction"].agg(["mean", "std"])
    eta_means["mean"].plot(kind="line", ax=ax, marker="o", yerr=eta_means["std"])
    ax.set_xlabel("eta_adv (adversarial noise)")
    ax.set_ylabel("Honest Reporting Fraction")
    ax.set_title("Prompt Injection Robustness")
    ax.set_ylim(0, 1)

    plt.tight_layout()
    output_file = output_dir / "figure6_prompt_injection.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "figure6_prompt_injection.pdf", bbox_inches="tight")
    logger.info(f"Saved Figure 6 to {output_file}")


def generate_figure7(df: pd.DataFrame, output_dir: Path):
    """Generate Figure 7: Sybil/Reputation Dynamics."""
    if "n_sybil" not in df.columns:
        logger.warning("Skipping Figure 7: results file has no n_sybil column")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    sybil_means = df.groupby("n_sybil")["honest_reporting_fraction"].agg(["mean", "std"])
    sybil_means["mean"].plot(kind="bar", ax=ax, yerr=sybil_means["std"], capsize=3, color="red")
    ax.set_xlabel("Number of Sybil Validators")
    ax.set_ylabel("Honest Reporting Fraction")
    ax.set_title("Sybil Attack Resistance")
    ax.set_ylim(0, 1)

    ax = axes[1]
    if {"round", "reputation_mean_honest", "reputation_mean_sybil"}.issubset(df.columns):
        rep_means = df.groupby("round")[["reputation_mean_honest", "reputation_mean_sybil"]].mean()
        rep_means.plot(ax=ax)
        ax.set_xlabel("Round")
        ax.set_ylabel("Mean Reputation")
        ax.set_title("Reputation Dynamics")
    else:
        trimmed_means = df.groupby("n_sybil")["sybil_trimmed"].mean()
        trimmed_means.plot(kind="bar", ax=ax, color="purple")
        ax.set_xlabel("Number of Sybil Validators")
        ax.set_ylabel("Mean Trimmed Submissions")
        ax.set_title("Sybil Trimming")

    plt.tight_layout()
    output_file = output_dir / "figure7_sybil_reputation.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "figure7_sybil_reputation.pdf", bbox_inches="tight")
    logger.info(f"Saved Figure 7 to {output_file}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate figures from experiment results")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/runs",
        help="Directory containing results",
    )
    parser.add_argument(
        "--results-file",
        type=str,
        default="results.csv",
        help="Results CSV file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="figures",
        help="Output directory for figures",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = data_dir / args.results_file
    if not results_file.exists():
        logger.error(f"Results file not found: {results_file}")
        return

    logger.info(f"Loading results from {results_file}")
    df = pd.read_csv(results_file)
    logger.info(f"Loaded {len(df)} rows")

    logger.info("Generating figures...")

    generate_figure3(df, output_dir)
    generate_figure4(df, output_dir)

    if "model" in df.columns and df["model"].nunique() > 1:
        generate_figure5(df, output_dir)

    generate_figure6(df, output_dir)
    generate_figure7(df, output_dir)

    logger.info(f"\nAll figures saved to {output_dir}")


if __name__ == "__main__":
    main()
