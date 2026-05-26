#!/usr/bin/env python3
"""Run prompt-injection robustness experiments for Figure 6."""

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


def load_attacks(path: str) -> list[dict]:
    attacks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                attacks.append(json.loads(line))
    if not attacks:
        raise ValueError(f"No attacks found in {path}")
    return attacks


def run_cell(loader, asset, model, seed, tau, lambda_, attack, n_queries, n_validators):
    queries = loader.generate_queries(asset, n_queries=n_queries, seed=seed)
    qre = LogitQRESampler(grid_size=1024, tau=tau, lambda_=lambda_)
    qre.set_seed(seed)
    oracle = FabricOracleClient()

    aggregate_predictions = []
    ground_truth_bins = []
    latencies = []

    for query in queries:
        injected_context = query.news_context
        if attack["attack_text"]:
            injected_context = f"{query.news_context}\n\nPotentially adversarial context:\n{attack['attack_text']}"

        oracle.submit_query(
            query_id=query.query_id,
            asset=asset,
            ground_truth=query.ground_truth_settlement,
            bin_id=query.bin_id,
        )
        for validator_idx in range(n_validators):
            prior = query.prior_7day_window["Close"].values
            if attack["eta_adv"] > 0 and validator_idx == 0:
                if "1023" in attack["attack_text"]:
                    pred_bin = 1023
                elif '"bin": 0' in attack["attack_text"]:
                    pred_bin = 0
                elif '"bin": 512' in attack["attack_text"]:
                    pred_bin = 512
                else:
                    pred_bin = qre.sample(prior, price_domain=query.price_domain)
            else:
                pred_bin = qre.sample(prior, price_domain=query.price_domain)
            oracle.submit_prediction(
                query_id=query.query_id,
                validator_id=f"validator_{validator_idx}_{seed}",
                predicted_bin=pred_bin,
                confidence=max(0.1, 1.0 - float(attack["eta_adv"]) * 0.2),
            )
        settlement = oracle.settle(query.query_id, lambda_param=lambda_)
        aggregate_predictions.append(int(settlement.aggregated_bin))
        ground_truth_bins.append(int(query.bin_id))
        latencies.append(settlement.commit_latency_ms)

    pred_arr = np.asarray(aggregate_predictions, dtype=float)
    truth_arr = np.asarray(ground_truth_bins, dtype=float)
    return {
        "asset": asset,
        "model": model,
        "method": "stylised_prompt_injection",
        "seed": seed,
        "tau": tau,
        "lambda": lambda_,
        "attack_id": attack["attack_id"],
        "eta_adv": float(attack["eta_adv"]),
        "attack_description": attack.get("description", ""),
        "n_queries": len(queries),
        "n_validators": n_validators,
        "honest_reporting_fraction": float(np.mean(pred_arr == truth_arr)),
        "welfare_gap": float(np.mean(np.abs(pred_arr - truth_arr) / 1024)),
        "parse_failures": 0,
        "avg_commit_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Run prompt-injection robustness experiment")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="data/runs/prompt_injection")
    parser.add_argument("--attacks-file", default="config/prompt_injection_attacks.jsonl")
    parser.add_argument("--assets", nargs="+", default=["XAU_USD"])
    parser.add_argument("--model", default="stylised_qre")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1234, 2345, 3456, 4567, 5678])
    parser.add_argument("--tau", type=float, default=0.7)
    parser.add_argument("--lambda-value", type=float, default=0.5)
    parser.add_argument("--n-queries", type=int, default=1000)
    parser.add_argument("--n-validators", type=int, default=64)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    loader = AssetLoader(data_dir=args.data_dir)
    attacks = load_attacks(args.attacks_file)

    rows = []
    for asset in args.assets:
        for seed in args.seeds:
            for attack in attacks:
                logger.info("Running asset=%s seed=%s attack=%s", asset, seed, attack["attack_id"])
                rows.append(
                    run_cell(
                        loader=loader,
                        asset=asset,
                        model=args.model,
                        seed=seed,
                        tau=args.tau,
                        lambda_=args.lambda_value,
                        attack=attack,
                        n_queries=args.n_queries,
                        n_validators=args.n_validators,
                    )
                )

    output_file = output_dir / "prompt_injection_results.csv"
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "generated_at": datetime.now().isoformat(),
        "output_file": str(output_file),
        "attacks_file": args.attacks_file,
        "n_rows": len(rows),
        "note": "Stylised local prompt-injection robustness run. Use LLM-backed runner before claiming actual LLM robustness.",
    }
    with open(output_dir / "prompt_injection_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Saved prompt-injection results to %s", output_file)


if __name__ == "__main__":
    main()
