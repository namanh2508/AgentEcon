"""
AgentEcon Experiment Runner

Configurable experiment runner for reproducibility:
- YAML/JSON configuration
- Multiple modes: LLM, Stylised QRE, or mixed
- Full logging with deterministic seeds and hashes
- Progressive scaling: smoke test -> pilot -> mini grid -> full grid
"""

import csv
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.data.asset_loader import AssetLoader, SettlementQuery
from src.models.predictors import (
    LLMPredictor,
    LogitQRESampler,
    PredictionResult,
    PromptTemplate,
)
from src.oracle.fabric_oracle import FabricOracleClient, SettlementResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _json_default(value):
    """Convert NumPy scalar/array values emitted by experiments into JSON."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


@dataclass
class ExperimentConfig:
    """Configuration for an experiment run."""

    experiment_name: str
    mode: str = "llm"

    models: list[str] = field(
        default_factory=lambda: ["meta-llama/Meta-Llama-3.1-8B-Instruct"]
    )
    assets: list[str] = field(default_factory=lambda: ["XAU_USD"])
    seeds: list[int] = field(default_factory=lambda: [1234])

    tau_values: list[float] = field(default_factory=lambda: [0.5, 0.7, 1.0, 1.5, 2.0])
    lambda_values: list[float] = field(default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0])

    n_validators: int = 64
    n_llm_validators: int = 4
    n_stylised_validators: int = 60
    n_queries: int = 1000
    n_queries_pilot: int = 200
    n_queries_smoke: int = 10
    calibration_queries: int = 200

    grid_size: int = 1024

    enable_sybil_detection: bool = True
    enable_reputation: bool = True
    sybil_ratio: float = 0.0

    use_stylised_for_large: bool = False
    stylised_tv_threshold: float = 0.04
    llm_batch_size: int = 8
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.85
    max_model_len: int = 2048
    hf_cache_dir: str = ""
    news_enabled: bool = False
    fabric_backend: str = "simulator"

    output_dir: str = "data/runs"

    def __post_init__(self):
        valid_modes = {"llm", "stylised", "mixed"}
        if self.mode not in valid_modes:
            raise ValueError(f"mode must be one of {sorted(valid_modes)}, got {self.mode!r}")
        if self.n_validators <= 0:
            raise ValueError("n_validators must be positive")
        if self.n_llm_validators < 0 or self.n_stylised_validators < 0:
            raise ValueError("validator counts cannot be negative")
        if self.mode == "mixed":
            expected = self.n_llm_validators + self.n_stylised_validators
            if expected != self.n_validators:
                raise ValueError(
                    "mixed mode requires n_llm_validators + n_stylised_validators "
                    f"to equal n_validators ({expected} != {self.n_validators})"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_name": self.experiment_name,
            "mode": self.mode,
            "models": self.models,
            "assets": self.assets,
            "seeds": self.seeds,
            "tau_values": self.tau_values,
            "lambda_values": self.lambda_values,
            "n_validators": self.n_validators,
            "n_llm_validators": self.n_llm_validators,
            "n_stylised_validators": self.n_stylised_validators,
            "n_queries": self.n_queries,
            "n_queries_pilot": self.n_queries_pilot,
            "n_queries_smoke": self.n_queries_smoke,
            "calibration_queries": self.calibration_queries,
            "grid_size": self.grid_size,
            "enable_sybil_detection": self.enable_sybil_detection,
            "enable_reputation": self.enable_reputation,
            "sybil_ratio": self.sybil_ratio,
            "use_stylised_for_large": self.use_stylised_for_large,
            "stylised_tv_threshold": self.stylised_tv_threshold,
            "llm_batch_size": self.llm_batch_size,
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_model_len": self.max_model_len,
            "hf_cache_dir": self.hf_cache_dir,
            "news_enabled": self.news_enabled,
            "fabric_backend": self.fabric_backend,
            "output_dir": self.output_dir,
        }

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    @classmethod
    def from_json(cls, path: str) -> "ExperimentConfig":
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


@dataclass
class ExperimentRun:
    """Record of a single experiment run."""

    run_id: str
    config: ExperimentConfig
    start_time: str
    end_time: Optional[str] = None
    git_hash: str = ""
    model_revision: str = ""
    dataset_hash: str = ""
    prompt_hash: str = ""
    status: str = "running"
    metrics: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


class ExperimentRunner:
    """
    Main experiment runner for AgentEcon.

    Supports multiple execution modes:
    - `llm`: Use actual LLM inference via vLLM
    - `stylised`: Use QRE sampler only
    - `mixed`: LLM for calibration, stylised for large-scale

    Progressive scaling:
    1. Smoke test: n_queries_smoke (10) queries
    2. Pilot: n_queries_pilot (200) queries
    3. Mini grid: 1 asset × 1 model
    4. Full grid: all configurations
    """

    SEEDS = [1234, 2345, 3456, 4567, 5678]

    def __init__(
        self,
        config: ExperimentConfig,
        data_dir: str = "data",
        asset_loader: Optional[AssetLoader] = None,
        oracle_client: Optional[FabricOracleClient] = None,
    ):
        self.config = config
        self.data_dir = Path(data_dir)
        configured_output = Path(config.output_dir)
        if configured_output.is_absolute() or configured_output.parts[:1] == ("data",):
            self.output_dir = configured_output
        else:
            self.output_dir = self.data_dir / configured_output
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.asset_loader = asset_loader or AssetLoader(data_dir=str(self.data_dir))
        self.oracle = oracle_client or FabricOracleClient()

        self.current_run: Optional[ExperimentRun] = None
        self._llm_predictors: dict[str, LLMPredictor] = {}

    def _get_git_hash(self) -> str:
        """Get current git commit hash."""
        try:
            import subprocess

            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                cwd=Path.cwd(),
            )
            return result.stdout.strip()[:16]
        except Exception:
            return "unknown"

    def _get_llm_predictor(self, model_name: str) -> LLMPredictor:
        """Get or create LLM predictor for a model."""
        if model_name not in self._llm_predictors:
            self._llm_predictors[model_name] = LLMPredictor(
                model_name=model_name,
                tensor_parallel_size=self.config.tensor_parallel_size,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                max_model_len=self.config.max_model_len,
            )
        return self._llm_predictors[model_name]

    def _predict_with_llm(self, predictor: LLMPredictor, asset: str, q: SettlementQuery) -> PredictionResult:
        """Run one LLM prediction for a settlement query."""
        return predictor.predict(
            query_id=q.query_id,
            asset_name=self.asset_loader.ASSET_CONFIGS[asset]["name"],
            query_date=(q.timestamp - pd.Timedelta(days=7)).strftime("%Y-%m-%d"),
            settlement_date=q.timestamp.strftime("%Y-%m-%d"),
            prior_7day_df=q.prior_7day_window,
            min_price=q.price_domain[0],
            max_price=q.price_domain[1],
            grid_size=self.config.grid_size,
            news_context=q.news_context,
        )

    def _predict_with_qre(self, qre: LogitQRESampler, q: SettlementQuery) -> PredictionResult:
        """Run one stylised QRE prediction for a settlement query."""
        start_time = time.time()
        prior = q.prior_7day_window["Close"].values
        pred_bin = qre.sample(prior, price_domain=q.price_domain)
        latency_ms = (time.time() - start_time) * 1000
        return PredictionResult(
            query_id=q.query_id,
            predicted_bin=pred_bin,
            confidence=0.5,
            raw_output="stylised",
            parse_success=True,
            latency_ms=latency_ms,
            model_name="stylised_qre",
        )

    def _generate_run_id(self) -> str:
        """Generate unique run ID from config hash."""
        config_str = json.dumps(self.config.to_dict(), sort_keys=True)
        hash_str = hashlib.sha256(config_str.encode()).hexdigest()[:12]
        return f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash_str}"

    def _setup_logging(self, run_id: str):
        """Setup file logging for this run."""
        log_file = self.output_dir / f"{run_id}.log"
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        return log_file

    def _log_row(self, csv_writer, row: dict):
        """Write a row to the CSV log."""
        csv_writer.writerow(row)

    def _compute_honest_reporting_fraction(
        self,
        predictions: list[int],
        ground_truth: int | list[int],
        tolerance: int = 0,
    ) -> float:
        """Compute fraction of predictions within tolerance of ground truth."""
        if not predictions:
            return 0.0
        pred_arr = np.asarray(predictions, dtype=float)
        truth_arr = np.asarray(ground_truth, dtype=float)
        if truth_arr.ndim == 0:
            truth_arr = np.full_like(pred_arr, float(truth_arr))
        if len(pred_arr) != len(truth_arr):
            raise ValueError("predictions and ground_truth must have equal length")
        return float(np.mean(np.abs(pred_arr - truth_arr) <= tolerance))

    def _compute_welfare_gap(
        self,
        predictions: list[int],
        ground_truth: int | list[int],
        grid_size: int = 1024,
    ) -> float:
        """Compute average welfare gap (normalized distance to ground truth)."""
        if not predictions:
            return 0.0
        pred_arr = np.asarray(predictions, dtype=float)
        truth_arr = np.asarray(ground_truth, dtype=float)
        if truth_arr.ndim == 0:
            truth_arr = np.full_like(pred_arr, float(truth_arr))
        if len(pred_arr) != len(truth_arr):
            raise ValueError("predictions and ground_truth must have equal length")
        return float(np.mean(np.abs(pred_arr - truth_arr) / grid_size))

    def run_smoke_test(self) -> dict[str, Any]:
        """
        Run smoke test: verify all components work.

        Returns:
            dict with smoke test results
        """
        logger.info("=== Running smoke test ===")
        results = {"status": "pass", "errors": []}

        try:
            for asset in ["XAU_USD", "EUR_USD"]:
                queries = self.asset_loader.generate_queries(
                    asset, n_queries=5, seed=1234
                )
                if len(queries) < 5:
                    results["errors"].append(f"Failed to generate queries for {asset}")

            if self.config.mode in ["llm", "mixed"]:
                predictor = self._get_llm_predictor(self.config.models[0])

            oracle = FabricOracleClient()
            oracle.submit_query(
                query_id="smoke_0",
                asset="XAU_USD",
                ground_truth=1850.50,
                bin_id=512,
            )
            for i in range(3):
                oracle.submit_prediction(
                    query_id="smoke_0",
                    validator_id=f"validator_{i}",
                    predicted_bin=512,
                )
            result = oracle.settle(query_id="smoke_0", lambda_param=0.5)
            logger.info(f"Smoke test settlement: {result}")

            qre = LogitQRESampler(grid_size=1024, tau=1.0)
            for seed in self.SEEDS[:2]:
                qre.set_seed(seed)
                sample = qre.sample(np.array([100.0, 101.0, 102.0]))
                logger.debug(f"QRE sample (seed={seed}): bin={sample}")

        except Exception as e:
            results["status"] = "fail"
            results["errors"].append(str(e))
            logger.error(f"Smoke test failed: {e}")

        logger.info(f"Smoke test: {results['status']}")
        return results

    def run_pilot(
        self,
        asset: str,
        model: str,
        seed: int,
        n_queries: int = 200,
    ) -> dict[str, Any]:
        """
        Run pilot: measure throughput and calibrate stylised sampler.

        Args:
            asset: Asset to use
            model: Model name
            seed: Random seed
            n_queries: Number of queries

        Returns:
            dict with pilot metrics
        """
        logger.info(f"=== Running pilot: {asset}, {model}, seed={seed} ===")

        queries = self.asset_loader.generate_queries(asset, n_queries=n_queries, seed=seed)

        prompt_template = PromptTemplate()
        prompt_hash = prompt_template.hash_prompt()

        results = {
            "asset": asset,
            "model": model,
            "seed": seed,
            "n_queries": len(queries),
            "prompt_hash": prompt_hash,
            "predictions": [],
            "latencies": [],
            "parse_failures": 0,
        }

        effective_mode = self.config.mode
        if effective_mode in ["llm", "mixed"]:
            predictor = self._get_llm_predictor(model)
            try:
                predictor.initialize()
            except Exception as e:
                logger.warning(f"LLM initialization failed: {e}, using stylised")
                effective_mode = "stylised"

        qre = LogitQRESampler(
            grid_size=self.config.grid_size,
            tau=self.config.tau_values[0],
            lambda_=self.config.lambda_values[0],
        )
        qre.set_seed(seed)

        start_time = time.time()

        for q in queries:
            if effective_mode == "stylised":
                result = self._predict_with_qre(qre, q)
            else:
                result = self._predict_with_llm(predictor, asset, q)

            results["predictions"].append(result.predicted_bin)
            results["latencies"].append(result.latency_ms)
            if not result.parse_success:
                results["parse_failures"] += 1

        total_time = time.time() - start_time
        avg_latency = np.mean(results["latencies"]) if results["latencies"] else 0
        throughput = len(queries) / total_time if total_time > 0 else 0

        ground_truth_bins = [q.bin_id for q in queries]
        honest_frac = self._compute_honest_reporting_fraction(
            results["predictions"], ground_truth_bins
        )
        welfare_gap = self._compute_welfare_gap(results["predictions"], ground_truth_bins)

        results.update(
            {
                "total_time_s": total_time,
                "avg_latency_ms": avg_latency,
                "throughput_qps": throughput,
                "honest_reporting_fraction": honest_frac,
                "welfare_gap": welfare_gap,
            }
        )

        logger.info(
            f"Pilot complete: {len(queries)} queries in {total_time:.1f}s, "
            f"avg_latency={avg_latency:.1f}ms, honest={honest_frac:.3f}"
        )

        return results

    def run_cell(
        self,
        asset: str,
        model: str,
        seed: int,
        tau: float,
        lambda_: float,
        n_queries: int,
        mode: str = "stylised",
    ) -> dict[str, Any]:
        """
        Run a single experiment cell.

        A cell is one configuration of (asset, model, seed, tau, lambda).

        Args:
            asset: Asset key
            model: Model name
            seed: Random seed
            tau: QRE temperature parameter
            lambda_: Reputation weighting parameter
            n_queries: Number of queries
            mode: 'llm', 'stylised', or 'mixed'

        Returns:
            dict with cell results
        """
        cell_id = f"{asset}_{model}_{seed}_tau{tau}_lam{lambda_}"
        logger.info(f"Running cell: {cell_id}")

        queries = self.asset_loader.generate_queries(asset, n_queries=n_queries, seed=seed)

        qre = LogitQRESampler(
            grid_size=self.config.grid_size,
            tau=tau,
            lambda_=lambda_,
        )
        qre.set_seed(seed)

        aggregate_predictions = []
        submitted_predictions = []
        latencies = []
        llm_latencies = []
        stylised_latencies = []
        parse_failures = 0

        llm_predictor = None
        if mode in {"llm", "mixed"}:
            llm_predictor = self._get_llm_predictor(model)

        oracle = FabricOracleClient()

        for q in queries:
            oracle.submit_query(
                query_id=q.query_id,
                asset=asset,
                ground_truth=q.ground_truth_settlement,
                bin_id=q.bin_id,
            )

            for v in range(self.config.n_validators):
                validator_id = f"validator_{v}_{seed}"

                if mode == "llm":
                    result = self._predict_with_llm(llm_predictor, asset, q)
                    llm_latencies.append(result.latency_ms)
                elif mode == "mixed" and v < self.config.n_llm_validators:
                    result = self._predict_with_llm(llm_predictor, asset, q)
                    llm_latencies.append(result.latency_ms)
                else:
                    result = self._predict_with_qre(qre, q)
                    stylised_latencies.append(result.latency_ms)

                if not result.parse_success:
                    parse_failures += 1

                oracle.submit_prediction(
                    query_id=q.query_id,
                    validator_id=validator_id,
                    predicted_bin=result.predicted_bin,
                    confidence=result.confidence,
                )
                submitted_predictions.append(int(result.predicted_bin))

            settlement = oracle.settle(q.query_id, lambda_param=lambda_)

            aggregate_predictions.append(int(settlement.aggregated_bin))
            latencies.append(settlement.commit_latency_ms)

        ground_truth_bins = [int(q.bin_id) for q in queries]
        honest_frac = self._compute_honest_reporting_fraction(aggregate_predictions, ground_truth_bins)
        welfare_gap = self._compute_welfare_gap(aggregate_predictions, ground_truth_bins)
        expanded_truth_bins = np.repeat(ground_truth_bins, self.config.n_validators).tolist()
        validator_honest_frac = self._compute_honest_reporting_fraction(
            submitted_predictions,
            expanded_truth_bins,
        )
        validator_welfare_gap = self._compute_welfare_gap(
            submitted_predictions,
            expanded_truth_bins,
        )

        return {
            "cell_id": cell_id,
            "asset": asset,
            "model": model,
            "method": mode,
            "seed": seed,
            "tau": tau,
            "lambda": lambda_,
            "n_queries": len(queries),
            "n_validators": self.config.n_validators,
            "n_llm_validators": (
                self.config.n_validators if mode == "llm"
                else self.config.n_llm_validators if mode == "mixed"
                else 0
            ),
            "n_stylised_validators": (
                0 if mode == "llm"
                else self.config.n_stylised_validators if mode == "mixed"
                else self.config.n_validators
            ),
            "honest_reporting_fraction": honest_frac,
            "welfare_gap": welfare_gap,
            "validator_honest_reporting_fraction": validator_honest_frac,
            "validator_welfare_gap": validator_welfare_gap,
            "parse_failures": parse_failures,
            "avg_llm_latency_ms": float(np.mean(llm_latencies)) if llm_latencies else 0.0,
            "avg_stylised_latency_ms": float(np.mean(stylised_latencies)) if stylised_latencies else 0.0,
            "avg_commit_latency_ms": float(np.mean(latencies)),
            "predictions": aggregate_predictions,
            "ground_truth_bins": ground_truth_bins,
        }

    def run_full_grid(self, progress_callback=None) -> list[dict]:
        """
        Run full experiment grid.

        Returns:
            List of cell results
        """
        logger.info("=== Running full experiment grid ===")
        logger.info(
            f"Grid: {len(self.config.assets)} assets × {len(self.config.models)} models × "
            f"{len(self.config.tau_values)} tau × {len(self.config.lambda_values)} lambda × "
            f"{len(self.config.seeds)} seeds = "
            f"{len(self.config.assets) * len(self.config.models) * len(self.config.tau_values) * len(self.config.lambda_values) * len(self.config.seeds)} cells"
        )

        results = []
        total_cells = (
            len(self.config.assets)
            * len(self.config.models)
            * len(self.config.tau_values)
            * len(self.config.lambda_values)
            * len(self.config.seeds)
        )
        cell_count = 0

        for asset in self.config.assets:
            for model in self.config.models:
                for tau in self.config.tau_values:
                    for lambda_ in self.config.lambda_values:
                        for seed in self.config.seeds:
                            cell_result = self.run_cell(
                                asset=asset,
                                model=model,
                                seed=seed,
                                tau=tau,
                                lambda_=lambda_,
                                n_queries=self.config.n_queries,
                                mode=self.config.mode,
                            )
                            results.append(cell_result)
                            cell_count += 1

                            if progress_callback:
                                progress_callback(cell_count, total_cells)

                            if cell_count % 10 == 0:
                                logger.info(
                                    f"Progress: {cell_count}/{total_cells} cells "
                                    f"({100 * cell_count / total_cells:.1f}%)"
                                )

        logger.info(f"Full grid complete: {len(results)} cells")
        return results

    def save_results(self, results: list[dict], filename: str = "results.csv"):
        """Save results to CSV with manifest."""
        output_file = self.output_dir / filename
        manifest_file = self.output_dir / f"{filename}_manifest.json"

        if not results:
            logger.warning("No results to save")
            return

        fieldnames = [
            "cell_id",
            "asset",
            "model",
            "method",
            "seed",
            "tau",
            "lambda",
            "n_queries",
            "n_validators",
            "n_llm_validators",
            "n_stylised_validators",
            "honest_reporting_fraction",
            "welfare_gap",
            "validator_honest_reporting_fraction",
            "validator_welfare_gap",
            "parse_failures",
            "avg_llm_latency_ms",
            "avg_stylised_latency_ms",
            "avg_commit_latency_ms",
        ]

        with open(output_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for r in results:
                row = {k: r.get(k) for k in fieldnames}
                self._log_row(writer, row)

        manifest = {
            "output_file": str(output_file),
            "n_cells": len(results),
            "config": self.config.to_dict(),
            "run_id": self.current_run.run_id if self.current_run else "unknown",
            "git_hash": self.current_run.git_hash if self.current_run else "unknown",
            "timestamp": datetime.now().isoformat(),
        }

        with open(manifest_file, "w") as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"Results saved to {output_file}")

    def run(
        self,
        scale: str = "full",
        save_results: bool = True,
    ) -> dict[str, Any]:
        """
        Run experiment with specified scale.

        Args:
            scale: 'smoke', 'pilot', 'mini', or 'full'
            save_results: Whether to save results to disk

        Returns:
            dict with experiment summary
        """
        run_id = self._generate_run_id()
        self.current_run = ExperimentRun(
            run_id=run_id,
            config=self.config,
            start_time=datetime.now().isoformat(),
            git_hash=self._get_git_hash(),
        )

        log_file = self._setup_logging(run_id)
        logger.info(f"Starting experiment: {run_id}")
        logger.info(f"Config: {json.dumps(self.config.to_dict(), indent=2)}")

        summary = {"run_id": run_id, "scale": scale, "status": "running"}

        try:
            if scale == "smoke":
                smoke_results = self.run_smoke_test()
                summary["smoke_test"] = smoke_results
                summary["status"] = "complete" if smoke_results["status"] == "pass" else "failed"

            elif scale == "pilot":
                pilot_results = self.run_pilot(
                    asset=self.config.assets[0],
                    model=self.config.models[0],
                    seed=self.config.seeds[0],
                    n_queries=self.config.n_queries_pilot,
                )
                summary["pilot"] = pilot_results
                summary["status"] = "complete"

            elif scale == "mini":
                cell_result = self.run_cell(
                    asset=self.config.assets[0],
                    model=self.config.models[0],
                    seed=self.config.seeds[0],
                    tau=self.config.tau_values[0],
                    lambda_=self.config.lambda_values[0],
                    n_queries=self.config.n_queries_pilot,
                    mode=self.config.mode,
                )
                summary["mini_grid"] = [cell_result]
                summary["status"] = "complete"

            elif scale == "full":
                grid_results = self.run_full_grid()
                summary["full_grid"] = grid_results
                summary["n_cells"] = len(grid_results)

                if save_results:
                    self.save_results(grid_results, f"{run_id}_results.csv")

                honest_fracs = [r["honest_reporting_fraction"] for r in grid_results]
                welfare_gaps = [r["welfare_gap"] for r in grid_results]
                summary["summary"] = {
                    "mean_honest_fraction": float(np.mean(honest_fracs)),
                    "std_honest_fraction": float(np.std(honest_fracs)),
                    "mean_welfare_gap": float(np.mean(welfare_gaps)),
                    "std_welfare_gap": float(np.std(welfare_gaps)),
                }
                summary["status"] = "complete"

        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            summary["status"] = "failed"
            summary["error"] = str(e)
            self.current_run.errors.append(str(e))

        self.current_run.end_time = datetime.now().isoformat()
        self.current_run.status = summary["status"]

        summary_file = self.output_dir / f"{run_id}_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2, default=_json_default)

        logger.info(f"Experiment {run_id} {summary['status']}")
        logger.info(f"Results: {log_file}")

        return summary
if __name__ == "__main__":
    config = ExperimentConfig(
        experiment_name="test_run",
        mode="stylised",
        assets=["XAU_USD"],
        models=["meta-llama/Meta-Llama-3.1-8B-Instruct"],
        seeds=[1234],
        tau_values=[0.5, 1.0],
        lambda_values=[0.0, 0.5],
        n_validators=4,
        n_queries=100,
    )

    runner = ExperimentRunner(config, data_dir="data")
    summary = runner.run(scale="smoke")
    print(f"Smoke test: {summary}")
