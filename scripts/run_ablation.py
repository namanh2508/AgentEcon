#!/usr/bin/env python3
"""Run oracle mechanism ablations for Table III."""

import argparse
import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data.asset_loader import AssetLoader
from src.models.predictors import LogitQRESampler
from src.oracle.fabric_oracle import FabricOracleClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


ABLATIONS = [
    {
        "config_name": "full",
        "use_trimmed_mean": True,
        "use_reputation": True,
        "use_sybil_detection": True,
        "use_qsr": True,
    },
    {
        "config_name": "no_trimmed_mean",
        "use_trimmed_mean": False,
        "use_reputation": True,
        "use_sybil_detection": True,
        "use_qsr": True,
    },
    {
        "config_name": "no_reputation",
        "use_trimmed_mean": True,
        "use_reputation": False,
        "use_sybil_detection": True,
        "use_qsr": True,
    },
    {
        "config_name": "no_sybil_detection",
        "use_trimmed_mean": True,
        "use_reputation": True,
        "use_sybil_detection": False,
        "use_qsr": True,
    },
    {
        "config_name": "no_qsr_confidence",
        "use_trimmed_mean": True,
        "use_reputation": True,
        "use_sybil_detection": True,
        "use_qsr": False,
    },
]


def run_ablation_case(loader, asset, seed, config, n_queries, n_validators, tau, lambda_):
    queries = loader.generate_queries(asset, n_queries=n_queries, seed=seed)
    qre = LogitQRESampler(grid_size=1024, tau=tau, lambda_=lambda_)
    qre.set_seed(seed)
    oracle = FabricOracleClient()
    if not config["use_trimmed_mean"]:
        oracle.aggregator.trim_fraction = 0.0

    predictions = []
    truths = []
    latencies = []
    trimmed = []

    for query in queries:
        oracle.submit_query(query.query_id, asset, query.ground_truth_settlement, query.bin_id)
        for validator_idx in range(n_validators):
            validator_id = f"validator_{validator_idx}_{seed}"
            if not config["use_reputation"]:
                oracle.reputation.reputations[validator_id] = 1.0
            pred_bin = qre.sample(query.prior_7day_window["Close"].values, price_domain=query.price_domain)
            confidence = 1.0 if config["use_qsr"] else 0.5
            oracle.submit_prediction(query.query_id, validator_id, pred_bin, confidence=confidence)

        settlement = oracle.settle(
            query.query_id,
            lambda_param=lambda_ if config["use_reputation"] else 0.0,
            sybil_detection=config["use_sybil_detection"],
        )
        predictions.append(int(settlement.aggregated_bin))
        truths.append(int(query.bin_id))
        latencies.append(float(settlement.commit_latency_ms))
        trimmed.append(int(settlement.sybil_trimmed))

    pred_arr = np.asarray(predictions, dtype=float)
    truth_arr = np.asarray(truths, dtype=float)
    return {
        **config,
        "asset": asset,
        "seed": seed,
        "tau": tau,
        "lambda": lambda_,
        "n_queries": len(queries),
        "n_validators": n_validators,
        "honest_reporting_fraction": float(np.mean(pred_arr == truth_arr)),
        "welfare_gap": float(np.mean(np.abs(pred_arr - truth_arr) / 1024)),
        "avg_commit_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
        "avg_trimmed": float(np.mean(trimmed)) if trimmed else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Run oracle ablations")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="data/runs/ablation")
    parser.add_argument("--asset", default="XAU_USD")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1234, 2345, 3456, 4567, 5678])
    parser.add_argument("--n-queries", type=int, default=1000)
    parser.add_argument("--n-validators", type=int, default=64)
    parser.add_argument("--tau", type=float, default=0.7)
    parser.add_argument("--lambda-value", type=float, default=0.5)
    args = parser.parse_args()

    loader = AssetLoader(data_dir=args.data_dir)
    rows = []
    for seed in args.seeds:
        for config in ABLATIONS:
            logger.info("Running seed=%s ablation=%s", seed, config["config_name"])
            rows.append(
                run_ablation_case(
                    loader=loader,
                    asset=args.asset,
                    seed=seed,
                    config=config,
                    n_queries=args.n_queries,
                    n_validators=args.n_validators,
                    tau=args.tau,
                    lambda_=args.lambda_value,
                )
            )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "ablation_results.csv"
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with open(output_dir / "ablation_manifest.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at": datetime.now().isoformat(),
                "output_file": str(output_file),
                "n_rows": len(rows),
                "ablation_configs": ABLATIONS,
            },
            f,
            indent=2,
        )
    logger.info("Saved ablation results to %s", output_file)


if __name__ == "__main__":
    main()
