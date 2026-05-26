"""
AgentEcon - Reproducible Economic Agent Oracle System

A system for replicating the paper results on LLM-based prediction markets
with Hyperledger Fabric oracle and QRE equilibrium analysis.
"""

__version__ = "1.0.0"
__author__ = "AgentEcon Research Team"

from src.data.asset_loader import AssetLoader
from src.models.predictors import LogitQRESampler
from src.oracle.fabric_oracle import FabricOracleClient
from src.experiments.runner import ExperimentRunner

__all__ = [
    "AssetLoader",
    "LogitQRESampler",
    "FabricOracleClient",
    "ExperimentRunner",
]
