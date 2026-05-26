#!/usr/bin/env python3
"""
Analyze experiment results and generate statistical report.
"""

import logging
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analysis.statistics import StatisticalAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/runs",
        help="Directory containing experiment results",
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
        default=None,
        help="Output directory for report",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=10000,
        help="Number of bootstrap samples",
    )

    args = parser.parse_args()

    data_dir = args.data_dir
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = data_dir

    logger.info("=== Statistical Analysis ===")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Results file: {args.results_file}")
    logger.info(f"Bootstrap samples: {args.n_bootstrap}")

    analyzer = StatisticalAnalyzer(
        data_dir=data_dir,
        n_bootstrap=args.n_bootstrap,
    )

    results_file = Path(data_dir) / args.results_file
    if not results_file.exists():
        logger.error(f"Results file not found: {results_file}")
        logger.info("Available files:")
        for f in Path(data_dir).glob("*.csv"):
            logger.info(f"  {f.name}")
        return

    df = analyzer.load_results(args.results_file)
    logger.info(f"Loaded {len(df)} experiment cells")

    report = analyzer.generate_report(df, output_dir=output_dir)

    logger.info("\n=== Summary ===")
    logger.info(f"Overall honest reporting: {report['honest_reporting']['overall_mean']:.4f}")
    logger.info(f"Overall welfare gap: {report['welfare_gap']['overall_mean']:.4f}")
    logger.info(f"Mean commit latency: {report['commit_latency']['overall_mean_ms']:.2f}ms")

    if "figure3" in report:
        logger.info("\nFigure 3 (Honest Reporting) Analysis:")
        for tau in report["figure3"].get("tau_effects", []):
            logger.info(f"  tau={tau['tau']}: {tau['mean']:.4f} ± {tau['std']:.4f}")

    if "figure4" in report:
        logger.info("\nFigure 4 (Welfare Gap) Analysis:")
        for item in report["figure4"].get("lambda_welfare", []):
            logger.info(f"  lambda={item['lambda']}: gap={item['mean_welfare_gap']:.4f}")

    if "figure5" in report:
        logger.info("\nFigure 5 (Multi-LLM) Analysis:")
        if "model_means" in report["figure5"]:
            model_means = report["figure5"]["model_means"]
            mean_by_model = model_means.get("mean", {}) if isinstance(model_means, dict) else {}
            for model, mean_value in mean_by_model.items():
                try:
                    logger.info(f"  {model}: {float(mean_value):.4f}")
                except (TypeError, ValueError):
                    logger.info(f"  {model}: {mean_value}")

    report_file = Path(output_dir) / "statistical_report.json"
    logger.info(f"\nFull report saved to: {report_file}")


if __name__ == "__main__":
    main()
