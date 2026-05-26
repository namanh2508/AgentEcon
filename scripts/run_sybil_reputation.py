#!/usr/bin/env python3
"""Run Sybil/reputation dynamics experiments for Figure 7."""

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


def run_sybil_case(loader, asset, seed, n_sybil, n_rounds, n_validators, tau, lambda_):
    queries = loader.generate_queries(asset, n_queries=n_rounds, seed=seed)
    qre = LogitQRESampler(grid_size=1024, tau=tau, lambda_=lambda_)
    qre.set_seed(seed)
    oracle = FabricOracleClient()

    honest_count = max(0, n_validators - n_sybil)
    sybil_ids = [f"sybil_{idx}_{seed}" for idx in range(n_sybil)]
    honest_ids = [f"honest_{idx}_{seed}" for idx in range(honest_count)]
    for validator_id in sybil_ids:
        oracle.reputation.reputations[validator_id] = 0.05

    rows = []
    for round_idx, query in enumerate(queries, start=1):
        oracle.submit_query(
            query_id=query.query_id,
            asset=asset,
            ground_truth=query.ground_truth_settlement,
            bin_id=query.bin_id,
        )

        for validator_id in honest_ids:
            pred_bin = qre.sample(query.prior_7day_window["Close"].values, price_domain=query.price_domain)
            oracle.submit_prediction(query.query_id, validator_id, pred_bin, confidence=0.7)

        sybil_target = 1023 - int(query.bin_id)
        for validator_id in sybil_ids:
            oracle.submit_prediction(query.query_id, validator_id, sybil_target, confidence=1.0)

        settlement = oracle.settle(query.query_id, lambda_param=lambda_, sybil_detection=True)

        honest_reps = [oracle.get_reputation(v) for v in honest_ids] or [0.0]
        sybil_reps = [oracle.get_reputation(v) for v in sybil_ids] or [0.0]
        rows.append(
            {
                "asset": asset,
                "seed": seed,
                "round": round_idx,
                "tau": tau,
                "lambda": lambda_,
                "n_sybil": n_sybil,
                "sybil_ratio": n_sybil / n_validators if n_validators else 0.0,
                "n_validators": n_validators,
                "honest_reporting_fraction": float(settlement.honest_reporting),
                "welfare_gap": float(settlement.welfare_gap),
                "reputation_mean_honest": float(np.mean(honest_reps)),
                "reputation_mean_sybil": float(np.mean(sybil_reps)),
                "sybil_trimmed": int(settlement.sybil_trimmed),
                "avg_commit_latency_ms": float(settlement.commit_latency_ms),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description="Run Sybil/reputation dynamics")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="data/runs/sybil_reputation")
    parser.add_argument("--asset", default="XAU_USD")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1234, 2345, 3456, 4567, 5678])
    parser.add_argument("--n-sybil-values", nargs="+", type=int, default=[0, 8, 16])
    parser.add_argument("--n-rounds", type=int, default=120)
    parser.add_argument("--n-validators", type=int, default=64)
    parser.add_argument("--tau", type=float, default=0.7)
    parser.add_argument("--lambda-value", type=float, default=0.5)
    args = parser.parse_args()

    loader = AssetLoader(data_dir=args.data_dir)
    rows = []
    for seed in args.seeds:
        for n_sybil in args.n_sybil_values:
            logger.info("Running seed=%s n_sybil=%s", seed, n_sybil)
            rows.extend(
                run_sybil_case(
                    loader=loader,
                    asset=args.asset,
                    seed=seed,
                    n_sybil=n_sybil,
                    n_rounds=args.n_rounds,
                    n_validators=args.n_validators,
                    tau=args.tau,
                    lambda_=args.lambda_value,
                )
            )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "sybil_reputation_results.csv"
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with open(output_dir / "sybil_reputation_manifest.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at": datetime.now().isoformat(),
                "output_file": str(output_file),
                "n_rows": len(rows),
            },
            f,
            indent=2,
        )
    logger.info("Saved Sybil/reputation results to %s", output_file)


if __name__ == "__main__":
    main()
