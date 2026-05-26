#!/usr/bin/env python3
"""Smoke test script for AgentEcon."""

import sys
sys.path.insert(0, ".")

# Test imports
from src.data.asset_loader import AssetLoader, SettlementQuery
print("[OK] AssetLoader")

from src.models.predictors import LogitQRESampler, PromptTemplate, JSONOutputParser
print("[OK] Models")

from src.oracle.fabric_oracle import FabricOracleClient, QuadraticScoringRule
print("[OK] Oracle")

from src.experiments.runner import ExperimentConfig, ExperimentRunner
print("[OK] ExperimentRunner")

from src.analysis.statistics import BootstrapCI, StatisticalAnalyzer
print("[OK] Statistics")

# Test QRE Sampler
import numpy as np
sampler = LogitQRESampler(grid_size=1024, tau=1.0)
sampler.set_seed(1234)
sample = sampler.sample(np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0]))
print(f"[OK] QRE sample: bin={sample}")

# Test Oracle
oracle = FabricOracleClient()
oracle.submit_query("test_q1", "XAU_USD", 1850.50, 512)
oracle.submit_prediction("test_q1", "validator_0", 500)
result = oracle.settle("test_q1")
print(f"[OK] Settlement: honest={result.honest_reporting}")

# Test Parser
parser = JSONOutputParser()
bin_id, conf, success = parser.parse('{"bin": 512, "confidence": 0.75}')
print(f"[OK] Parser: bin={bin_id}, conf={conf}, success={success}")

# Test Reputation Tracker
from src.oracle.fabric_oracle import ReputationTracker
tracker = ReputationTracker()
rep = tracker.get_reputation("new_validator")
print(f"[OK] Reputation: {rep}")

# Test Bootstrap CI
bs = BootstrapCI(n_bootstrap=100)
x = np.random.randn(100)
y = np.random.randn(100)
stat, ci_low, ci_high = bs.paired_ci(x, y)
print(f"[OK] Bootstrap CI: stat={stat:.4f}, CI=[{ci_low:.4f}, {ci_high:.4f}]")

print()
print("=" * 50)
print("All smoke tests PASSED!")
print("=" * 50)
