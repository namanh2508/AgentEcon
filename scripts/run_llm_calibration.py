#!/usr/bin/env python3
"""
Run the Week-1 LLM/QRE calibration protocol.

This script measures actual LLM predictions on a held-out query subset and
calibrates the Stylised Logit-QRE sampler against those predictions. By
default it requires vLLM and local/accessible Hugging Face model weights.
"""

import argparse
import csv
import hashlib
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data.asset_loader import AssetLoader
from src.models.predictors import LLMPredictor, LogitQRESampler, PromptTemplate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


DEFAULT_MODELS = [
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "Qwen/Qwen2.5-7B-Instruct",
]

DEFAULT_ASSETS = ["XAU_USD", "WTI", "EUR_USD", "CASE_SHILLER"]


def _git_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip()[:16] or "unknown"
    except Exception:
        return "unknown"


def _dataset_hash(queries) -> str:
    payload = [query.dataset_hash for query in queries]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def _safe_name(value: str) -> str:
    return value.replace("/", "__").replace(":", "_")


def _predict_with_llm(predictor: LLMPredictor, loader: AssetLoader, asset: str, query):
    return predictor.predict(
        query_id=query.query_id,
        asset_name=loader.ASSET_CONFIGS[asset]["name"],
        query_date=(query.timestamp - pd.Timedelta(days=7)).strftime("%Y-%m-%d"),
        settlement_date=query.timestamp.strftime("%Y-%m-%d"),
        prior_7day_df=query.prior_7day_window,
        min_price=query.price_domain[0],
        max_price=query.price_domain[1],
        grid_size=query.grid_size,
        news_context=query.news_context,
    )


def run_calibration_cell(
    *,
    loader: AssetLoader,
    asset: str,
    model: str,
    seed: int,
    n_queries: int,
    target_tv: float,
    dry_run_stylised: bool,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    max_model_len: int,
):
    queries = loader.generate_queries(asset, n_queries=n_queries, seed=seed)
    if not queries:
        raise RuntimeError(f"No queries generated for asset={asset}, seed={seed}")

    predictor = None
    if not dry_run_stylised:
        predictor = LLMPredictor(
            model_name=model,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
        )
        predictor.initialize()

    qre_for_dry_run = LogitQRESampler(grid_size=loader.GRID_SIZE, tau=0.7, lambda_=0.5)
    qre_for_dry_run.set_seed(seed)

    prediction_rows = []
    llm_bins = []
    latencies = []
    parse_failures = 0

    for query in queries:
        if dry_run_stylised:
            prior = query.prior_7day_window["Close"].values
            predicted_bin = qre_for_dry_run.sample(prior, price_domain=query.price_domain)
            confidence = 0.5
            parse_success = True
            latency_ms = 0.0
            raw_output = "stylised_dry_run"
        else:
            result = _predict_with_llm(predictor, loader, asset, query)
            predicted_bin = result.predicted_bin
            confidence = result.confidence
            parse_success = result.parse_success
            latency_ms = result.latency_ms
            raw_output = result.raw_output

        llm_bins.append(int(predicted_bin))
        latencies.append(float(latency_ms))
        if not parse_success:
            parse_failures += 1

        prediction_rows.append(
            {
                "asset": asset,
                "model": model,
                "seed": seed,
                "query_id": query.query_id,
                "ground_truth_bin": int(query.bin_id),
                "predicted_bin": int(predicted_bin),
                "confidence": float(confidence),
                "parse_success": bool(parse_success),
                "latency_ms": float(latency_ms),
                "raw_output": raw_output[:500],
            }
        )

    calibration_qre = LogitQRESampler(grid_size=loader.GRID_SIZE, tau=0.7, lambda_=0.5)
    eta0, achieved_tv = calibration_qre.calibrate_eta0(
        llm_bins=np.asarray(llm_bins, dtype=int),
        prior_data=[query.prior_7day_window["Close"].values for query in queries],
        price_domains=[query.price_domain for query in queries],
        target_tv=target_tv,
    )

    summary = {
        "asset": asset,
        "model": model,
        "seed": seed,
        "n_queries": len(queries),
        "calibration_source": "stylised_dry_run" if dry_run_stylised else "vllm",
        "eta0": float(eta0),
        "tv_distance": float(achieved_tv),
        "target_tv": float(target_tv),
        "is_reportable_stylised_qre": bool((not dry_run_stylised) and achieved_tv <= target_tv),
        "parse_failures": int(parse_failures),
        "parse_failure_rate": float(parse_failures / len(queries)),
        "avg_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
        "p50_latency_ms": float(np.percentile(latencies, 50)) if latencies else 0.0,
        "p95_latency_ms": float(np.percentile(latencies, 95)) if latencies else 0.0,
        "dataset_hash": _dataset_hash(queries),
        "prompt_hash": PromptTemplate().hash_prompt(),
    }
    return summary, prediction_rows


def main():
    parser = argparse.ArgumentParser(description="Run LLM/QRE calibration")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="data/runs/calibration")
    parser.add_argument("--assets", nargs="+", default=DEFAULT_ASSETS)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--n-queries", type=int, default=200)
    parser.add_argument("--target-tv", type=float, default=0.04)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument(
        "--dry-run-stylised",
        action="store_true",
        help="Use stylised predictions instead of vLLM. This is for pipeline testing only and is not protocol-valid.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = AssetLoader(data_dir=args.data_dir)
    summary_rows = []
    all_prediction_rows = []

    for asset in args.assets:
        for model in args.models:
            logger.info("Calibrating asset=%s model=%s seed=%s", asset, model, args.seed)
            summary, prediction_rows = run_calibration_cell(
                loader=loader,
                asset=asset,
                model=model,
                seed=args.seed,
                n_queries=args.n_queries,
                target_tv=args.target_tv,
                dry_run_stylised=args.dry_run_stylised,
                tensor_parallel_size=args.tensor_parallel_size,
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_model_len=args.max_model_len,
            )
            summary_rows.append(summary)
            all_prediction_rows.extend(prediction_rows)
            logger.info(
                "Done asset=%s model=%s eta0=%.4f TV=%.4f reportable=%s",
                asset,
                model,
                summary["eta0"],
                summary["tv_distance"],
                summary["is_reportable_stylised_qre"],
            )

    summary_file = output_dir / "calibration_results.csv"
    prediction_file = output_dir / "calibration_predictions.csv"
    manifest_file = output_dir / "calibration_manifest.json"

    with open(summary_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    with open(prediction_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_prediction_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_prediction_rows)

    manifest = {
        "generated_at": datetime.now().isoformat(),
        "git_hash": _git_hash(),
        "data_dir": args.data_dir,
        "output_dir": str(output_dir),
        "summary_file": str(summary_file),
        "prediction_file": str(prediction_file),
        "assets": args.assets,
        "models": args.models,
        "seed": args.seed,
        "n_queries": args.n_queries,
        "target_tv": args.target_tv,
        "dry_run_stylised": args.dry_run_stylised,
        "protocol_valid": not args.dry_run_stylised,
    }
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Calibration summary saved to %s", summary_file)
    logger.info("Calibration predictions saved to %s", prediction_file)
    logger.info("Calibration manifest saved to %s", manifest_file)


if __name__ == "__main__":
    main()
